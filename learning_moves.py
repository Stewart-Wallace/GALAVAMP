from random import random, choice
from math import gamma,comb
from abc import ABC, abstractmethod
from json import load
from os import getcwd
from scipy.stats import gamma as scipy_gamma
import sys
import numpy as np
import config_file

def new_operator(operators,flavour:str,exc:list):
    exc = [e[1:] for e in exc if e[1] == flavour]
    possible_strings = [s for s in list(operators.keys()) if s not in exc]
    if flavour == 'H':
        op_string = choice([s for s in possible_strings])
    if flavour == 'L':
        op_string = choice(list(possible_strings))
    return flavour+op_string



class Move(ABC):
    '''
    Move is the parent class for any move within model space. The initialisation
    parses a probability of the move taking place, for use in the likelihood and
    the current particle that is being changed. These objects should take the 
    below structure:
        1) change: a value between 0 and 1, corresponding to the likelihood of
            this move's selection.
        2) current_particle: A dictionary of string terms as keys and float
            values as values, corresponding to the rates of those terms.
    '''
    def __init__(self,learning_params,operators, verbose = False):
        self.learning_params = learning_params
        self.verbose = verbose

    @abstractmethod
    def Enact(self,operators,current_particle):
        self.trial_particle = current_particle.copy()
        pass
    

class PARAMETER(Move):

    def Enact(self,operators,current_particle):
        self.current_particle = current_particle
        self.trial_particle = self.current_particle.copy()
        op = choice(list(self.trial_particle.keys()))
        val = self.trial_particle[op]
        if op == 'bkg':
            self.trial_particle[op] = np.abs(np.random.normal(val,self.learning_params['initial_variance']/10))
        else:
            self.trial_particle[op] = np.abs(np.random.normal(val,self.learning_params['initial_variance']))
        return self.trial_particle
    
class ADDINGH(Move):

    def Enact(self, operators,current_particle):
        self.current_particle = current_particle
        self.trial_particle = self.current_particle.copy()
        op = new_operator(operators[0],'H',self.current_particle.keys())
        self.new_rate = np.random.gamma(config_file.learning_params['parameter_proposal_shape'],scale=config_file.learning_params['parameter_proposal_scale'])
        self.trial_particle[op] = self.new_rate
        return self.trial_particle
    
class REMOVINGH(Move):

    def Enact(self, operators,current_particle):
        self.current_particle = current_particle
        if len([t for t in list(self.current_particle.keys()) if t.startswith('H')]) > 0:
            self.trial_particle = self.current_particle.copy()
            op = choice([t for t in list(self.trial_particle.keys()) if t.startswith('H')])
            self.new_rate = self.trial_particle[op]
            del self.trial_particle[op]
            return self.trial_particle
        else:
            return None
    
class SWAPPINGH(Move):

    def Enact(self,operators,current_particle):
        self.current_particle = current_particle
        if len([t for t in list(self.current_particle.keys()) if t.startswith('H')]) > 0:
            self.trial_particle = self.current_particle.copy()
            op_new = new_operator(operators[0],'H',self.current_particle.keys())
            op_old = choice([t for t in list(self.trial_particle.keys()) if t.startswith('H')])
            del self.trial_particle[op_old]
            self.trial_particle[op_new] = np.random.gamma(config_file.learning_params['parameter_proposal_shape'],scale=config_file.learning_params['parameter_proposal_scale'])
            return self.trial_particle
        else:
            return None
    
class ADDINGL_SINGLE(Move):

    def Enact(self, operators,current_particle):
        self.current_particle = current_particle
        self.trial_particle = self.current_particle.copy()
        op = new_operator(operators[1],'L',self.current_particle.keys())
        self.new_rate = np.random.gamma(config_file.learning_params['parameter_proposal_shape'],scale=config_file.learning_params['parameter_proposal_scale'])
        self.trial_particle[op] = self.new_rate
        return self.trial_particle

class REMOVINGL_SINGLE(Move):

    def Enact(self, operators,current_particle):
        self.current_particle = current_particle
    
        if len([t for t in list(self.current_particle.keys()) if t.startswith('L')]) > 0:
            self.trial_particle = self.current_particle.copy()
            op = choice([t for t in list(self.trial_particle.keys()) if t.startswith('L')])
            self.new_rate = self.trial_particle[op]
            del self.trial_particle[op]
            return self.trial_particle
        else:
            return None
    
class SWAPPINGL_SINGLE(Move):

    def Enact(self,operators,current_particle):
        self.current_particle = current_particle
        if len([t for t in list(self.current_particle.keys()) if t.startswith('L')]) > 0:
            self.trial_particle = self.current_particle.copy()
            op_new = new_operator(operators[1],'L',self.current_particle.keys())
            op_old = choice([t for t in list(self.trial_particle.keys()) if t.startswith('L')])
            del self.trial_particle[op_old]
            self.trial_particle[op_new] = np.random.gamma(config_file.learning_params['parameter_proposal_shape'],scale=config_file.learning_params['parameter_proposal_scale'])
            return self.trial_particle
        else:
            return None
    
class NUTCRACKER(Move):

    def Enact(self,operators,current_particle):
        self.current_particle = current_particle.copy()
        complex_terms = [L for L in list(self.current_particle.keys()) if L.startswith('L') and '+' in L]
        if complex_terms == []:
            return None
        else:
            term = choice(complex_terms)
            value = self.current_particle[term]
            del self.current_particle[term]
            t1 = term.split('+')[0]
            t2 = 'L'+term.split('+')[1]
            self.current_particle[t1] = value
            self.current_particle[t2] = value
            return self.current_particle