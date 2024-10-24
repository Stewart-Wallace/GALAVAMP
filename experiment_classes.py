import numpy as np
import sys
from random import random,choices
from scipy.linalg import expm
from scipy.ndimage.filters import gaussian_filter1d
from itertools import product
from abc import ABC, abstractmethod

def weighted_unique_samples_1(population,weights,k):
    if k>len(population):
        sys.exit('Error: samples exceeed population')
    if len(population) != len(weights):
        sys.exit('Error: population, weight mismatch. Population:' + str(len(population)) + ' Weights:' + str(len(weights)))
    if k == len(population):
        return np.asarray(range(len(population)))
    out = np.zeros(k)
    weights = np.asarray(weights)
    population = np.asarray(population)

    for i in range(k):
        weights = weights/sum(weights)
        ind = choices(population=range(len(weights)),weights = weights)
        out[i] = population[ind]
        population = np.delete(population,ind)
        weights = np.delete(weights,ind)
    return(out)

def matrix_interpretation(matrix_dictionary:dict,number_of_sites):
    def lindblad_to_liouvillian(mtx):
        '''
        Taking in a single Lindbladian matrix, this function converts the 
        matrix to one operating in Liouvillian space following equation
        from arXiv:1804.11293v2
        '''
        
        term_1 = np.kron(mtx,mtx.conj())
        term_2 = -0.5 * ( np.kron( np.dot( mtx.conj().T , mtx) ,
                                    np.eye( int( np.sqrt(mtx.size)))))
        
        term_3 = -0.5 * ( np.kron( np.eye( int( np.sqrt(mtx.size))) ,
                                    np.dot( mtx.conj().T , mtx).T))
        return term_1 + term_2 + term_3                                     

    def hamiltonian_to_liouvillian(mtx):
        '''
        Taking in a Hamiltonian matrix and converts it to Liouville space.
        '''
        return -1j*( np.kron( mtx , np.eye( int( np.sqrt(mtx.size)))) 
                    - np.kron( np.eye( int( np.sqrt( mtx.size))) , mtx.T))
    
    
    output_mtx = np.zeros((2**(2*number_of_sites),2**(2*number_of_sites)), dtype = 'complex') 
    for term in matrix_dictionary.keys():
        if term.startswith('H'):
            output_mtx += hamiltonian_to_liouvillian(matrix_dictionary[term][0])*matrix_dictionary[term][1]
        elif term.startswith('L'):
            output_mtx += lindblad_to_liouvillian(matrix_dictionary[term][0])*matrix_dictionary[term][1]
        elif term == 'bkg':
            pass
        else:
            sys.exit('Term not recognised: '+ term)
    return output_mtx

def matricise(vec):
    '''
    Converts a vectorised matrix back t it's matrix form eg.:
    [1,2,3,4] --> [[1,3],[2,4]] (the inverse of np.flatten())
    '''
    import math
    N = int(np.sqrt(len(vec)))                                                  # Establish length size of square matrix from length of vector
    rho = np.zeros((N,N),complex)                                               # Intialises square matrix
    for c,val in enumerate(vec):                                                # Iterates through vector
        rho[math.floor(c/N),c%N] = val                                          # Constructs matrix
    return rho
    
class Experiment(ABC):
    '''
    Experiment is the parent class for any experimentation. The intialisation is
    standardised within the parent an d needs three agruments.
        1) number_of_sites: Number of 2 level systems, to inform the construction of
            the Liouvillian.
        2) operators: An exhaustive dictionary of operators considered in the search
            with strings (eg.'uu','xi+zx',...) as keys, and the corresponding 
            matrices as values.
        3) directory_of_data: This is the system directory that points to the 
            experimental data for the given calculation.
    '''
    def __init__(self,number_of_sites,operators,directory_of_data):
        self.number_of_sites = number_of_sites
        self.operators = operators
        with open(directory_of_data,'rb') as f:
            x = np.load(f)
            y = np.load(f)
        self.data = [x,y]
        self.importance_vector = np.abs(np.diff(gaussian_filter1d(self.data[1],sigma = 2, mode='nearest')))
        self.importance_vector = np.append(self.importance_vector,self.importance_vector[0])
        self.importance_vector = np.asarray(self.importance_vector) + np.mean(np.asarray(self.importance_vector))
        self.importance_vector = self.importance_vector/np.sum(self.importance_vector)
        self.sampling_steps = {
            0:1.0
        }
        self.resample_rate = 10
    
    @abstractmethod
    def Calculation(self,parameter_dictionary: dict,step):
        pass
    
    @abstractmethod
    def Experiment_Weighting(self,parameter_dictionary):
        pass

class Lifetime(Experiment):

    def Calculation(self,parameter_dictionary: dict,step):
        if step in self.sampling_steps.keys():
            self.sampling_proportion = self.sampling_steps[step]
            self.number_of_samples =  int(len(self.data[0])*self.sampling_proportion)
        if step%self.resample_rate == 0:
            self.indeces = weighted_unique_samples_1(np.asarray(range(len(self.data[0]))),weights = self.importance_vector,k =self.number_of_samples)
            self.indeces = sorted(self.indeces)
            self.sampled_data = [np.take(self.data[0],self.indeces),np.take(self.data[1],self.indeces)]
            self.chosen_weights = np.take(self.importance_vector,self.indeces)

        pumping_dictionary = {}
        emission_dictionary = {}
        misc_dictionary = {}
        matrix_dictionary = {k:self.operators[k[1:]] for k in parameter_dictionary.keys() if k !='bkg'}

        for term in matrix_dictionary.keys():
            mtx = matrix_dictionary[term]
            flavour = term[0]

            if flavour == 'H' and np.trace(mtx) == 0:
                pumping_dictionary[term] = [matrix_dictionary[term],parameter_dictionary[term]]
            elif flavour == 'L' and np.sum(np.abs(np.triu(mtx))) != 0 and np.trace(np.abs(mtx)) == 0:# and matrix_dictionary[term][2,1] == 0:
                pumping_dictionary[term] = [matrix_dictionary[term],parameter_dictionary[term]]
            elif flavour == 'L' and np.sum(np.abs(np.tril(mtx))) != 0 and np.trace(np.abs(mtx)) == 0:# and matrix_dictionary[term][1,2] == 0:
                emission_dictionary[term] = [matrix_dictionary[term],parameter_dictionary[term]]
            else:
                misc_dictionary[term] = [matrix_dictionary[term],parameter_dictionary[term]]

        excitation_mtx = matrix_interpretation(pumping_dictionary,self.number_of_sites)
        evolution_mtx = matrix_interpretation(emission_dictionary|misc_dictionary,self.number_of_sites)
        ##########
        # g2 distribution calculation
        ##########
        
        ground_state = np.zeros(((2*self.number_of_sites)**2,1))
        ground_state[-1] = 1
        fully_excited = np.zeros(((2*self.number_of_sites)**2,1))
        fully_excited[0] = 1
        
        vals = [[] for _ in range(len(emission_dictionary.keys())+1)]                                                                 # Initial list of g2 coincidence values from simulation
        for dd in self.sampled_data[0]:             # Iterates through times available from imported data.
            val_holder = 0
            evolved_exitation = matricise(np.dot(expm(evolution_mtx*dd),np.matrix(fully_excited)))
            for n,operator in enumerate(emission_dictionary.keys()):
                val_holder += emission_dictionary[operator][1]*emission_dictionary[operator][1]*np.trace(np.dot(np.dot(np.transpose(emission_dictionary[operator][0].conj()),emission_dictionary[operator][0]),evolved_exitation))
                vals[n].append(emission_dictionary[operator][1]*emission_dictionary[operator][1]*np.trace(np.dot(np.dot(np.transpose(emission_dictionary[operator][0].conj()),emission_dictionary[operator][0]),evolved_exitation)))
            vals[-1].append(val_holder)
        ##########
        # Convolve with Gaussian
        ##########c                                                                     # Convolve with a gaussian to replicate experimental conditions of detector jitter TODO generalise this.
        #vals = gaussian_filter1d(vals,sigma = 0.1019/(2*self.lim/self.integral_partitions), mode='constant', cval=vals[0])
        if type(max(vals[-1])) == int or max(vals[-1])==0j:
            vals[-1] = [np.complex128(1.) for _ in self.sampled_data[0]]
            #print('Zero Lifetime or integer maximum')

        vals[-1] = [v + parameter_dictionary['bkg'] for v in vals[-1]]
        temp = np.asarray(vals[-1]/max(vals[-1])).T.tolist() 
        #temp = [t+params[-1] for t in temp]
        #lizard print(np.sum(temp))
        if np.isnan(np.sum(temp)):
            print(matrix_interpretation(parameter_dictionary,self.number_of_sites))
        return np.real(temp)

    def Experiment_Weighting(self,parameter_dictionary,step):
        
        diff_hold = np.asarray(self.sampled_data[1])-np.asarray(self.Calculation(parameter_dictionary,step=1))          # Calculates the difference as a vector for application of weighting vector
        bars =  np.sum(np.dot(diff_hold.T,diff_hold)/self.sampling_proportion)                   # Calcuation of parameter for gamma distribution
        return np.real(np.random.gamma(len(self.data[0])/2 + 200000, scale = 1/(95.55 + bars/2)))  

class g2(Experiment):
    def __init__(self, number_of_sites, operators, directory_of_data):
        super().__init__(number_of_sites, operators, directory_of_data)
        self.importance_vector = np.real(np.asarray([-0.31720467 / ((-1.03063633) * (1 + (x / -1.03063633) ** 2)) + 0.049906940892216624  for x in self.data[0]])) # original: c = 0.049906940892216624


    def Calculation(self, parameter_dictionary: dict,step):
        '''
        This is one function that is most likley to be edited while changing the class.

        g2 distribution calulator takes variables as local variables not in the class 
        so it can be called for a trial particle and an intiial particle similarly.
        '''
        if step in self.sampling_steps.keys():
            self.sampling_proportion = self.sampling_steps[step]
            self.number_of_samples =  int(len(self.data[0])*self.sampling_proportion)
        if step%self.resample_rate == 0:
            self.indeces = weighted_unique_samples_1(np.asarray(range(len(self.data[0]))),weights = self.importance_vector,k =self.number_of_samples)
            self.indeces = sorted(self.indeces)
            self.sampled_data = [np.take(self.data[0],self.indeces),np.take(self.data[1],self.indeces)]
            self.chosen_weights = np.take(self.importance_vector,self.indeces)

        emission_dictionary = {}
        total_dictionary = {}
        matrix_dictionary = {k:self.operators[k[1:]] for k in parameter_dictionary.keys() if k !='bkg'}

        for term in matrix_dictionary.keys():
            if np.sum(np.abs(np.triu(matrix_dictionary[term])))==0 and np.sum(np.abs(np.tril(matrix_dictionary[term]))) != 0  and np.trace(matrix_dictionary[term]) == 0:# and matrix_dictionary[term][2,1] == 0:
                emission_dictionary[term] = [matrix_dictionary[term],parameter_dictionary[term]]
            total_dictionary[term] = [matrix_dictionary[term],parameter_dictionary[term]]                                               # Sequesters corresponding rates.
        evolution_mtx = matrix_interpretation(total_dictionary,self.number_of_sites)
        ##########
        # Temporary g2 calculations                                                                         
        ##########
        c_ss = np.dot(expm(evolution_mtx*(1000000+100*(random()))),np.full((np.shape(evolution_mtx)[0],1),1/np.shape(evolution_mtx)[0]))         # Calculates steady state of a given Liouvillian TODO replace with eigenvector method
        collective_emission_dictionary = {}
        if emission_dictionary == {}:
            emission_dictionary = {'LI':[np.eye(4),1.0]}
        for e in emission_dictionary.items():
            try:
                collective_emission_dictionary['New_operator'] = [collective_emission_dictionary['New_operator'][0]+e[1][0],1]
            except:
                collective_emission_dictionary['New_operator'] = [e[1][0],1]
        collective_emission_dictionary['New_operator'][1] = 1/np.sqrt(np.count_nonzero(collective_emission_dictionary['New_operator'][0]))
        #print('Collected Emission', collective_emission_dictionary)
        emission_dictionary = collective_emission_dictionary
        c_norm = 0                                                                      # Initialise normalisation value g2 construction TODO ask about validity here (also doesn't work)
        for term_0, term_1 in product(emission_dictionary.keys(),emission_dictionary.keys()):
            v_mtx   = emission_dictionary[term_0][0]
            u_mtx   = emission_dictionary[term_1][0]
            v_param = emission_dictionary[term_0][1]
            u_param = emission_dictionary[term_1][1]          
            c_norm += v_param*u_param*np.trace(np.dot(np.dot(np.transpose(u_mtx.conj()),u_mtx),matricise(np.dot(expm(evolution_mtx*np.abs(1000000)),np.dot(v_mtx,np.dot(matricise(c_ss),np.transpose(v_mtx.conj()))).flatten()))))
        c_norm += parameter_dictionary['bkg']                                                            # Adds background counts for the normalisation. TODO double check implementation of this
        
        ##########
        # g2 distribution calculation
        ##########
        vals = []                                                                       # Initial list of g2 coincidence values from simulation
        for dd in self.sampled_data[0]:             # Iterates through times available from imported data.
            val_holder = 0                                                              # Initialises placeholder value
            for term_0, term_1 in product(emission_dictionary.keys(),emission_dictionary.keys()):
                v_mtx   = emission_dictionary[term_0][0]
                u_mtx   = emission_dictionary[term_1][0]
                v_param = emission_dictionary[term_0][1]
                u_param = emission_dictionary[term_1][1]                # Then applies the calcualtion for a g2 for each combinationc, at each given time to calcualte the g2 at each tau.
                val_holder += v_param*u_param*np.trace(np.dot(np.dot(np.transpose(u_mtx.conj()),u_mtx),matricise(np.dot(expm(evolution_mtx*np.abs(dd)),np.dot(v_mtx,np.dot(matricise(c_ss),np.transpose(v_mtx.conj()))).flatten()))))/c_norm
            vals.append(val_holder+(parameter_dictionary['bkg']/c_norm))                                   # Accept a normalised g2 value TODO check that the normalisation is needed at this stage.
            
        ##########
        # Convolve with Gaussian
        ##########c                                                                     # Convolve with a gaussian to replicate experimental conditions of detector jitter TODO generalise this.
        vals = gaussian_filter1d(vals,sigma = 0.1019/(2*self.data[0][-1]/len(self.sampled_data[0])), mode='constant', cval=vals[0])
        if np.isnan(np.sum(vals)):
            print('NaNs in the water:', evolution_mtx)
        return np.real(vals)#/vals[0]  

    def Experiment_Weighting(self,parameter_dictionary,step):
        diff_hold = np.asarray(self.sampled_data[1])-np.asarray(self.Calculation(parameter_dictionary,step=1))          # Calculates the difference as a vector for application of weighting vector
        bars =  np.sum(np.dot(diff_hold.T,diff_hold)/self.sampling_proportion)                   # Calcuation of parameter for gamma distribution
        return np.real(np.random.gamma(len(self.data[0])/2 + 200000, scale = 1/(95.55 + bars/2)))  

