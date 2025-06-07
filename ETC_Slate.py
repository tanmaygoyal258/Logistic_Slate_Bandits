import numpy as np
from itertools import product
from tqdm import tqdm
from GLMEnv import GLMOracle
from utils import sigmoid

class ETC_Slate():

    def __init__(self ,  params , arm_set , theta_star):

        self.slot_count = params["slot_count"]
        self.item_count = params["item_count"]
        self.arm_dim = params["arm_dim"]
        self.horizon = params["horizon"]
        self.total_slates = self.item_count ** self.slot_count
        
        # we would like to deal with the combination of all indices rather than arms
        self.arm_indices = [[j for j in range(self.item_count)] for i in range(self.slot_count)]
        self.all_possible_indices = self.possible_combination_arm(self.arm_indices)

        self.arm_set = arm_set[0] # fixed arm setting
        self.all_possible_arms = self.possible_combination_arm(self.arm_set)

        self.oracle = GLMOracle(theta_star , sigmoid)

        self.m = 1
        self.kappa = (self.horizon**(-1/3)) * \
                    np.sqrt(self.item_count * np.log(self.horizon) * (1+self.m))
        self.gamma = 1/(self.horizon**self.m)
        self.N_hat = int(np.ceil(\
                    (2/(self.kappa**2) * (np.log(self.total_slates) - np.log(self.gamma)))\
                    ))
        self.V = [[[] for j in range(self.item_count)] for i in range(self.slot_count)]
        self.slate_samples = {tuple(slate) : 0 for slate in self.all_possible_indices}
        self.ctr = 0

        self.regret_arr = []
    
    def possible_combination_arm(self , arm_set):
        '''
        obtain possible combinations of arms
        '''
        all_combinations = list(product(*arm_set))
        return[np.array(arm).reshape(-1,) for arm in all_combinations]
    
    def best_arm_expected_reward(self):
        """
        Returns the expected reward of the best arm.
        """
        rewards = [self.oracle.expected_reward(arm) for arm in self.all_possible_arms]
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
    
    def explore_phase_1(self):
        print(f"Beginning Phase 1 of Exploration with length {self.item_count * self.N_hat}")
        for l in range(self.item_count):
            for n in range(self.N_hat):
                chosen_arm_indices = [l for _ in range(self.slot_count)]
                chosen_arm = np.hstack([self.arm_set[j][chosen_arm_indices[j]] for j in range(self.slot_count)])
                actual_reward , expected_regret , best_arm = self.interact(chosen_arm)
                
                # get samples for each of the slots where lth arm was pulled
                for m in range(self.slot_count):
                    padded_arm = [0 for _ in range(len(chosen_arm))]
                    dim = len(padded_arm)//self.slot_count
                    padded_arm[m*dim : (m+1)*dim] = self.arm_set[m][l]
                    expected_reward_linear = self.oracle.expected_reward_linear(padded_arm)                    
                    sample = expected_reward_linear + np.random.normal(0 , 1e-4)
                    self.V[m][l].append(sample)

                self.regret_arr.append(expected_regret)
                self.ctr += 1
                
    def explore_phase_2(self):
        print(f"Beginning Phase 2 of Exploration with length {self.total_slates * self.N_hat}")
        for slate in self.all_possible_indices:
            for n in range(self.N_hat):
                # pull the slate
                chosen_arm = np.hstack([self.arm_set[j][slate[j]] for j in range(self.slot_count)])
                actual_reward , expected_regret , best_arm = self.interact(chosen_arm)
                # store the samples
                samples_observed = [self.V[slot_num][arm_idx][n] for slot_num , arm_idx in enumerate(slate)]
                # assume function f is sigmoid on the addition of the samples
                self.slate_samples[tuple(slate)] += sigmoid(np.sum(samples_observed))

                self.regret_arr.append(expected_regret)
                self.ctr += 1
                if self.ctr >= self.horizon:
                    return

    def play_algorithm(self):
        self.explore_phase_1()
        self.explore_phase_2()

        if self.ctr < self.horizon:
            # find the best arm
            sorted_values = {k : v for k,v in sorted(self.slate_samples.items() , key = lambda item : item[1] , reverse=  True)}
            best_arm_indices = list(list(sorted_values.keys())[0])
            best_arm_chosen = np.hstack([self.arm_set[j][best_arm_indices[j]] for j in range(self.slot_count)])

            print("Commiting to the best arm")
            for _ in tqdm(range(self.ctr , self.horizon)):
                actual_reward , expected_regret , best_arm = self.interact(best_arm_chosen)
                self.regret_arr.append(expected_regret)

        assert len(self.regret_arr) == self.horizon , "Length of regret array is {}".format(len(self.regret_arr))
