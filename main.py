import sys
import inspect
import os
import numpy as np
import json
from datetime import date
from random import choices,random,sample,randint
from pathlib import Path
from itertools import product,permutations,combinations

import learning_moves
import experiment_classes
import config_file
import uniform_search

class learning_particle:
    def __init__(self,  experiment_list,                data_directory_list,
                        initial_particle = 'random',    number_of_sites = 2,
                        verbose = False):
        
        self.path = os.getcwd()
        self.number_of_sites = number_of_sites
        self.verbose = verbose
        self.initial_particle = initial_particle
        self.learning_params = config_file.learning_params.copy()
        self.save_list = []

        self.check_file_integrity()
        self.import_operators()
        uniform_search.uniform_search(dim = self.number_of_sites,verbose=self.verbose)

        self.experiments = []
        for e,d in zip(experiment_list,data_directory_list):
            exp_func_ = getattr(experiment_classes,e)
            self.experiments.append(exp_func_(self.number_of_sites,self.operators,d))
            if verbose:
                print("Experiment Loaded Correctly: \t\t" + e)
        
        self.accepted = 0
        self.move_weights = []
        self.MCMC_moves = {}

        for m in inspect.getmembers(learning_moves, inspect.isclass):
            if m[0] != 'ABC' and m[0] != 'Move':
                self.move_weights.append(self.learning_params[m[0]+'_chance'])
                print(self.verbose)
                self.MCMC_moves[m[0]] = m[1](self.learning_params,[self.H_ops,self.L_ops],verbose = self.verbose)
        self.move_weights = list(np.asarray(self.move_weights)/sum(self.move_weights))
        for n,k in enumerate(self.MCMC_moves.keys()):
            self.learning_params[k+'_chance'] = self.move_weights[n]

        self.cohort = []
        expected_parameters = int(self.learning_params['lindbladian_exponential_rate']+self.learning_params['hamiltonian_exponential_rate'])
        for _ in range(self.learning_params['cohort_size']):
            self.cohort.append([])
            self.cohort[-1].append(self.random_model(expected_parameters=expected_parameters))
            self.cohort[-1].append(self.evaluate(self.cohort[-1][0]))
            self.cohort[-1].append('Starter')

        self.cohort.sort(key=lambda x: x[1])
        self.top = []
        self.plus_std = []
        self.mean = []
        self.minus_std = []
        self.bot = []
    
    def update_data(self):
        self.save_list.append(self.cohort.copy())
        accuracies = np.asarray([c[1] for c in self.cohort])
        if self.verbose:
            print("###########################")
            print("\tTop Model: ", np.min(accuracies))
            print("\tAverage Kept: ", np.mean(accuracies[:int(self.learning_params['survival_ratio']*len(self.cohort))]))
            print("###########################")
        self.top.append(np.max(accuracies))
        self.bot.append(np.min(accuracies))
        self.mean.append(np.mean(accuracies))
        self.plus_std.append(np.mean(accuracies)+np.std(accuracies))
        self.minus_std.append(np.mean(accuracies)-np.std(accuracies))
   
    def evaluate(self,part):
        exp_results = [e.Calculation(part,step=0) for e in self.experiments]
        exp_diffs = [np.abs(res-exp.sampled_data[1]) for res,exp in zip(exp_results,self.experiments)]
        posteriors = [W/2*np.dot(d.T,np.multiply((e.chosen_weights),d)) for W,d,e in zip([20000,2000],exp_diffs,self.experiments)]
        return np.sum(posteriors)+self.learning_params['term_cost']*len(part.keys())+self.learning_params['complicated_term_cost']*len([i for i in list(part.keys()) if '+' in i])
    
    def check_file_integrity(self):
        chance_parameters = [k[:-7] for k in self.learning_params.keys() if k[-7:] == '_chance']
        move_functions = len(inspect.getmembers(learning_moves, inspect.isclass)) - 2

        if len(chance_parameters) > move_functions:
            sys.exit("From config_file.py there are " + str(len(chance_parameters)) +
                    " '_chance' parameters, and from learning_moves_py there are " +
                    str(move_functions) + " functions, this needs to match. Please" +
                    " remember to match names of functions to config chances.")
        missed = 0 
        for c in inspect.getmembers(learning_moves, inspect.isclass):
            if c[0] in chance_parameters:
                if self.verbose:
                    print("Learning Move Loaded Correctly:\t\t" + c[0])
            else:
                missed+=1
        if missed > 2:
            sys.exit("Name missmatch between config_file.py and learning_moves.py")

    def cull(self,n):
        survive =self.learning_params['survival_ratio']*len(self.cohort)
        new_len = int(survive+(survive*0.25*np.sin((n*5)/(2*np.pi))))
        if self.verbose:
            print("###########################")
            print('\tCull sine value: ',np.sin((n*5)/(2*np.pi)))
            print("\tNew cohort size: ", new_len)
            print("###########################")
        self.cohort = self.cohort[:new_len]

    def breed(self,part_1,part_2):
        total_genepool = part_1|part_2
        generated_part = {}
        generated_part['bkg'] = total_genepool['bkg']
        del total_genepool['bkg']
        for _ in range(randint(1,len(total_genepool.keys())/2)):
            k = choices(list(total_genepool.keys()))[0]
            generated_part[k] = total_genepool[k]
            del total_genepool[k]
        return(generated_part)
        
    def import_operators(self):
        op_path = (self.path  + r"/imag_opertators_"+str(self.number_of_sites)+
                                r"_complexity_"+str(self.learning_params['complexity'])+
                                r".npy")
        self.learning_params['operator path'] = op_path
        if os.path.isfile(op_path):
            if self.verbose:
                print("Operator Dictionary Loaded From: \t" + op_path)
                                
            numpy_ops = np.load(op_path, allow_pickle=True)
            self.operators = {k:v for k,v in zip(list(numpy_ops[0,:]),list(numpy_ops[1,:]))}

        else:
            if self.verbose:
                print("Constructing Operator Dictionary")

            def _lindblad_from_string(mtx_key,core_operator_dict):
                hold = np.zeros((2**self.number_of_sites,2**self.number_of_sites))
                mtx = 1
                for c in mtx_key:
                    if c == '+':
                        hold = mtx
                        mtx = 1
                    else:
                        mtx = np.kron(mtx,core_operator_dict[c])
                mtx = mtx + hold
                return np.where(np.real(mtx) == 0, 0, np.sign(np.real(mtx))) + 1j*np.where(np.imag(mtx) == 0, 0, np.sign(np.imag(mtx)))
                

            core_operator_dict = {
            'u': np.array([  # up
                [1 + 0.j, 0 + 0.j],
                [0 + 0.j, 0 + 0.j]
            ]),
            'd': np.array([  # down
                [0 + 0.j, 0 + 0.j],
                [0 + 0.j, 1 + 0.j]
            ]),
            'a': np.array([  # Add
                [0 + 0.j, 1 + 0.j],
                [0 + 0.j, 0 + 0.j]
            ]),
            's': np.array([  # Subtract
                [0 + 0.j, 0 + 0.j],
                [1 + 0.j, 0 + 0.j]
            ]),
            'U': np.array([  # up
                [0 + 1.j, 0 + 0.j],
                [0 + 0.j, 0 + 0.j]
            ]),
            'D': np.array([  # down
                [0 + 0.j, 0 + 0.j],
                [0 + 0.j, 0 + 1.j]
            ]),
            'A': np.array([  # Add
                [0 + 0.j, 0 + 1.j],
                [0 + 0.j, 0 + 0.j]
            ]),
            'S': np.array([  # Subtract
                [0 + 0.j, 0 + 0.j],
                [0 + 1.j, 0 + 0.j]
            ]),
            'W': np.array([  # up
                [0 - 1.j, 0 + 0.j],
                [0 + 0.j, 0 + 0.j]
            ]),
            'B': np.array([  # down
                [0 + 0.j, 0 + 0.j],
                [0 + 0.j, 0 - 1.j]
            ]),
            'Z': np.array([  # Add
                [0 + 0.j, 0 - 1.j],
                [0 + 0.j, 0 + 0.j]
            ]),
            'T': np.array([  # Subtract
                [0 + 0.j, 0 + 0.j],
                [0 - 1.j, 0 + 0.j]
            ])
            }
            if self.verbose:
                print("Building Operator Dictionary To: \t" + op_path)
                print("Operators Constructed With \t\t"+ str(list(core_operator_dict.keys())))
            self.operators = {}
            simple_terms = ["".join(s) for s in product(list(core_operator_dict.keys()),repeat = self.number_of_sites)]
            ##
            # Remove two photon processes
            ##
            for i in ['a','s','A','Z','S','T']:
                for j in ['a','s','A','Z','S','T']:
                    simple_terms.remove(i+j)
            print(simple_terms)

            string_terms = simple_terms + ["+".join(s) for s in permutations(simple_terms,self.learning_params['complexity'])]
            for n,k in enumerate(string_terms):
                mtx = _lindblad_from_string(k,core_operator_dict)
                if len([mtx for m in list(self.operators.values()) if np.allclose(mtx,m)]) == 0:
                    self.operators[k] = mtx

                if self.verbose and n%50 == 0 and n != 0:
                    print("Operators Explored: \t\t\t" + str(100*n/len(string_terms)) + r"%")
            if self.verbose:
                print("Unique Lindbladians Found: \t\t" + str(len(self.operators)))
            self.operators = {k:v for k,v in zip(list(self.operators.keys()),list(self.operators.values())) if (np.isreal(np.matrix(v)).all() or np.allclose(np.matrix(v).H, np.matrix(v))) and np.any(v)}
            np.save(op_path, np.array([list(self.operators.keys()),list(self.operators.values())], dtype=object), allow_pickle=True)

        self.L_ops = {k:v for k,v in zip(list(self.operators.keys()),list(self.operators.values())) if np.isreal(np.matrix(v)).all() and np.all(v>=0) and np.any(v)}
        self.H_ops = {k:v for k,v in zip(list(self.operators.keys()),list(self.operators.values())) if np.allclose(np.matrix(v).H, np.matrix(v)) and np.any(v)}

        print(self.H_ops)
        print(self.L_ops)

    def random_model(self,expected_parameters):
        adding_indexes = [n for n,k in enumerate(list(self.MCMC_moves.keys())) if k.startswith('ADDING')]
        adding_keys = [k for k in list(self.MCMC_moves.keys()) if k.startswith('ADDING')]
        temp_weights = [self.move_weights[i] for i in adding_indexes]
        new = {'bkg':np.random.gamma(2,scale=0.3)/10}
        while len(new.keys()) <= expected_parameters:
            move = choices(adding_keys,weights=temp_weights)[0]
            temp = self.MCMC_moves[move].Enact([self.H_ops,self.L_ops],new)
            if temp != None:
                new = temp
        return new
    
    def learning_loop(self):
        Path(self.learning_params['project_name']).mkdir(parents=True,exist_ok=True) 
        with open(self.learning_params['project_name']+'/data_' + str(date.today()) + '_' + 'config.json','w') as f:
            json.dump(self.learning_params,f)

        for s in range(1,self.learning_params['mutation_steps']):
            print('Learning Progression: ',(s/self.learning_params['mutation_steps'])*100,'%')
            self.cull(s)
            
            for n in range(len(self.cohort)):
                new = None
                while new == None:
                    new = self.MCMC_moves['PARAMETER'].Enact([self.H_ops,self.L_ops],self.cohort[n][0])
                    ev = self.evaluate(new)
                if self.cohort[n][1] > ev:
                    self.cohort[n][0] = new
                    self.cohort[n][1] = ev
                    print('Improvement:', n)
                
            while len(self.cohort) < self.learning_params['cohort_size']:
                R = random()
                if R < self.learning_params['random_rate']:
                    label = 'Random'
                    print(np.mean([len(c[0].keys()) for c in self.cohort]),np.std([len(c[0].keys()) for c in self.cohort]))
                    length = np.mean([len(c[0].keys()) for c in self.cohort]) + np.std([len(c[0].keys()) for c in self.cohort]) - 1
                    
                    new = self.random_model(expected_parameters=int(length))
                elif R < self.learning_params['mutation_rate']+ self.learning_params['random_rate']:
                    w = np.linspace(1,0,len(self.cohort))
                    i = choices(self.cohort,weights=w/np.sum(w))[0][0]
                    new = None
                    while new == None:
                        move = choices(list(self.MCMC_moves.keys()),weights=self.move_weights)[0]
                        new = self.MCMC_moves[move].Enact([self.H_ops,self.L_ops],i)
                    label = 'Mutation ' + move
                else: 
                    label = 'Bred'
                    new = None
                    c = 0
                    while new == None:
                        c+=1
                        w = np.linspace(1,0,len(self.cohort))
                        i = choices(self.cohort,weights=w/np.sum(w))[0][0]
                        j = choices(self.cohort,weights=w/np.sum(w))[0][0]
                        try:
                            new = self.breed(i,j)
                        except:
                            pass
                        if c> 5:
                            length = np.mean([len(c[0].keys()) for c in self.cohort]) + np.std([len(c[0].keys()) for c in self.cohort]) - 1
                            new = self.random_model(expected_parameters=int(length))
                self.cohort.append([new,self.evaluate(new)])
                self.cohort[-1].append(label)
            self.cohort.sort(key=lambda x: x[1])
            for c in self.cohort[:10]:
                print(c)
            self.update_data()
            print(len(list(set([''.join(sorted(list(p[0].keys())))+str(p[1]) for p in self.cohort])))/len([''.join(sorted(list(p[0].keys())))+str(p[1]) for p in self.cohort]),len(list(set([''.join(sorted(list(p[0].keys())))+str(p[1]) for p in self.cohort]))),len(list(set([''.join(sorted(list(p[0].keys()))) for p in self.cohort]))))

    def accept_particle(self,step,AR,log_likelihood,move, accepted):
        self.learning_data['step number'].append(step)
        self.learning_data['particle'].append(self.particle.copy())
        self.learning_data['posterior'].append(sum([self.current_exp_posterior][0]))
        self.learning_data['acceptance rate'].append(AR)
        self.learning_data['log likelihood'].append(log_likelihood)
        self.learning_data['experiment weighting'].append(self.current_exp_weights)
        self.learning_data['proposed move dict'][move].append(step)
        if accepted:
            self.learning_data['accepted move dict'][move].append(step)
    
    def save(self):
        Path(self.learning_params['project_name']).mkdir(parents=True,exist_ok=True)                            # Create location for file saving                                                                      
        with open(self.learning_params['project_name']+'/data_' + str(date.today()) + '_' + self.learning_params['project_name'] + str(int(100000000000*np.random.random())) + '.json', 'w') as f:         # Append name with random digit to make unique file
            try:
                json.dump(self.save_list,f)
            except:
                sys.exit('Save Failed',self.learning_data)


import matplotlib.pyplot as plt
data_name = [r"\both_dot_coherent.npy",r"\both_dot_lifetime_after_edit.npy"]
try:
    hi = learning_particle(['g2','Lifetime'],
            [r"C:\Users\sw2009\Documents\Python_Scripts\SCRIPTS\Cristian's Work"+data_name[0],
             r"C:\Users\sw2009\Documents\Python_Scripts\SCRIPTS\Cristian's Work"+data_name[1]],number_of_sites=2,verbose=True)
except:
    hi = learning_particle(['g2','Lifetime'],
        [r"/home/sw2009/python_scripts"+data_name[0],
        r"/home/sw2009/python_scripts"+data_name[1]],number_of_sites=2,verbose=True)
hi.learning_loop()
hi.save()

#plt.plot(hi.experiments[0].data[0],hi.experiments[0].Calculation(hi.cohort[0][0],step=1))
#plt.plot(hi.experiments[0].data[0],hi.experiments[0].Calculation(hi.cohort[1][0],step=1))
#plt.plot(hi.experiments[0].data[0],hi.experiments[0].data[1])
#plt.show()
#plt.plot(hi.experiments[1].data[0],hi.experiments[1].Calculation(hi.cohort[0][0],step=1))
#plt.plot(hi.experiments[1].data[0],hi.experiments[1].Calculation(hi.cohort[1][0],step=1))
#plt.plot(hi.experiments[1].data[0],hi.experiments[1].data[1])
#plt.show()
#plt.plot(hi.top)
#plt.plot(hi.plus_std)
#plt.plot(hi.mean)
#plt.plot(hi.minus_std)
#plt.plot(hi.bot)
#plt.show()



