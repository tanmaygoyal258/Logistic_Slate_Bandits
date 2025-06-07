import numpy as np
from tqdm import tqdm
from ada_OFU_ECOLog import ada_OFU_ECOLog
from Slate_GLM_OFU import Slate_GLM_OFU
from MPS import MPS
from TS_ECOLog import TS_ECOLog
from Slate_GLM_TS import Slate_GLM_TS
from itertools import product
from utils import sigmoid , probit , dsigmoid , dprobit
from time import time

class GLMOracle():
    def __init__(self , theta_star , reward_model):
        self.theta_star = theta_star
        self.reward_model = reward_model
    
    def expected_reward(self , arm):
        '''
        the expected reward is the sigmoid of the inner product between arm and optimal param
        '''
        return self.reward_model(np.dot(arm , self.theta_star))

    def expected_reward_linear(self , arm):
        '''
        the linear expected reward is the inner product between arm and optimal param
        '''
        return (np.dot(arm , self.theta_star))

    def pull(self , arm):
        '''
        the actual reward is sampled from a Bernoulli Distribution with mean equal to the expected reward
        '''
        return int(np.random.rand() < self.expected_reward(arm))
    

class GLMEnv():
    def __init__(self, params , slot_arms , theta_star):
        self.seed = params["seed"]

        self.alg_name = params["alg_name"]
        self.warmup = params["warmup"]
        self.reward_type = params["reward_type"]
        self.num_contexts = params["num_contexts"]       
        self.thetastar = theta_star
        self.reward_func = sigmoid if self.reward_type == "logistic" else probit
        self.d_reward_func = dsigmoid if self.reward_type == "logistic" else dprobit
        self.oracle = GLMOracle(theta_star , self.reward_func)

        self.item_count = params["item_count"]
        self.slot_count = params["slot_count"]
        self.horizon = params["horizon"]
        self.arm_dim = params["arm_dim"]
        self.param_norm_ub = params["param_norm_ub"]
        self.dim = self.slot_count * self.arm_dim
        
        self.regret_arr = np.empty(self.horizon)
        self.reward_arr = np.empty(self.horizon)
        self.pull_time_arr = np.empty(self.horizon)
        self.update_time_arr = np.empty(self.horizon)

        self.slot_arms = slot_arms
        self.all_combination_arms = []
        self.ctr = 0

        np.random.seed(self.seed)

        if self.alg_name == "ada_ofu_ecolog":
            self.alg = ada_OFU_ECOLog(params)
        elif self.alg_name == "slate_glm_ofu":
            self.alg = Slate_GLM_OFU(params)
        elif self.alg_name == "mps":
            self.alg = MPS(params)
        elif self.alg_name == "TS_ecolog":
            self.alg = TS_ECOLog(params)
        elif self.alg_name == "slate_glm_TS":
            self.alg = Slate_GLM_TS(params)
        elif self.alg_name == "slate_glm_TS_Fixed":
            arm_set = self.slot_arms[0] # fixed arm setting
            self.possible_combination_arms(arm_set)
            self.kappa = self.get_kappa()
            self.alg = Slate_GLM_TS(params , kappa = self.kappa , warmup = True) 
        else:
            assert False , "incorrect algorithm specified"

        self.eigenvalue_flag = "eigenvalues" in list(params.keys()) 

    def possible_combination_arms(self , arm_set):
        '''
        obtain possible combinations of arms
        '''
        self.all_combination_arms = list(product(*arm_set))
        self.all_combination_arms = [np.array(arm).reshape(-1,) for arm in self.all_combination_arms]
    
    
    def best_arm_expected_reward(self):
        """
        Returns the expected reward of the best arm.
        """
        rewards = [self.oracle.expected_reward(arm) for arm in self.all_combination_arms]
        best_arm = np.argmax(rewards)
        return rewards[best_arm] , best_arm

    def interact(self, arm):
        """
        Interacts with the environment to obtain the reward and instantaneous psuedo-regret.
        """
        expected_reward = self.oracle.expected_reward(arm)
        actual_reward = self.oracle.pull(arm)
        best_arm_reward , best_arm = self.best_arm_expected_reward()
        expected_regret = best_arm_reward - expected_reward
        return actual_reward, expected_regret , best_arm

    def play_algorithm(self):
        """
        Plays the algorithm on the environment.
        """

        for t in tqdm(range(self.horizon)):
            
            # obtain the arms
            arm_set = self.slot_arms[t] if self.num_contexts == self.horizon else self.slot_arms[np.random.choice(self.num_contexts)]

            # find all possible combinations of the arms
            self.possible_combination_arms(arm_set)

            # pull the arm
            if self.alg_name in ["ada_ofu_ecolog" , "TS_ecolog"]:
                pull_start = time()
                picked_embedding_indices = self.alg.pull(self.all_combination_arms)
                self.pull_time_arr[self.ctr] = time() - pull_start
                picked_embedding = self.all_combination_arms[picked_embedding_indices[0]]
            else:
                pull_start = time()
                picked_embedding_indices = self.alg.pull(arm_set)
                self.pull_time_arr[self.ctr] = time() - pull_start
                picked_arms = [arm_set[slot][i] for slot , i in enumerate(picked_embedding_indices)]
                picked_embedding = np.hstack(picked_arms)

            # obtain the actual reward and expected regret
            actual_reward , expected_regret , best_arm = self.interact(picked_embedding)

            # update the parameters
            update_start = time()
            self.alg.update_parameters(picked_embedding , actual_reward) if self.alg_name != "mps" else self.alg.update_parameters(picked_embedding_indices , actual_reward)
            self.update_time_arr[self.ctr] = time() - update_start

            # store the regrets, rewards, and time
            self.regret_arr[self.ctr] = expected_regret
            self.reward_arr[self.ctr] = actual_reward
            self.ctr += 1

        if self.eigenvalue_flag:
            self.eigenvalues = self.alg.eigenvalues

    def get_kappa(self):
        arm_set = self.all_combination_arms
        min_mu_dot = np.inf
        for arm in arm_set:
            min_mu_dot = min(min_mu_dot , self.d_reward_func(np.dot(arm , self.thetastar)))
        return 1.0/min_mu_dot