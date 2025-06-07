import numpy as np
import Non_Stochastic_Slate_Bandits.HelperAlgorithms as helpers
import Non_Stochastic_Slate_Bandits.MultiplicativeWeights as mw
from GLMEnv import GLMOracle
from itertools import product
from tqdm import tqdm
from utils import sigmoid , dsigmoid

class Ordered_Slot_Bandit():

    def __init__(self , params , arm_set , theta_star):
        self.alg_name = params["alg_name"]
        self.slot_count = params["slot_count"]
        self.item_count = params["item_count"]
        self.horizon = params["horizon"]
        self.total_arms = self.slot_count * self.item_count
        
        self.oracle = GLMOracle(theta_star , sigmoid)

        self.arm_set = arm_set[0] # fixed arm setting
        self.all_possible_arms = self.possible_combination_arm(self.arm_set)
        self.one_matrix_for_slots = self.construct_one_matrix_for_slots()

        self.regret_arr = []
        
        initial_dist = np.copy(self.one_matrix_for_slots) / self.total_arms
        self.gamma = np.sqrt(1.0 * self.total_arms * np.log(1.0*self.total_arms) / self.horizon)
        self.eta = np.sqrt((1.0-self.gamma) * np.log(1.0*self.total_arms)/(1.0*self.total_arms * self.horizon))
        self.mw_engine = mw.MultiplicativeWeights(initial_dist, self.slot_count, self.eta)

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
    
    def construct_one_matrix_for_slots(self):
        matrix = np.zeros((self.slot_count ,self.total_arms))
        for l in range(matrix.shape[0]):
            matrix[l , l*self.item_count : (l+1)*self.item_count] = 1
        return matrix


    def play_algorithm(self):

        for _ in tqdm(range(self.horizon)):
            current_distribution = np.copy(self.mw_engine.distribution)
            
            intermediate_dist = current_distribution * (1.0 - self.gamma) + (1.0*self.gamma / (self.total_arms) ) * self.one_matrix_for_slots 
            
            # flatten distribution
            flattened_dist = intermediate_dist.flatten(order='C')

            list_of_slate_prob_pairs = helpers.mixture_decomposition_ordered_stateless(self.slot_count, self.total_arms * self.slot_count, flattened_dist  )
            # repeat if it fails
            while (list_of_slate_prob_pairs == False):
                list_of_slate_prob_pairs = helpers.mixture_decomposition_ordered_stateless(self.slot_count, self.total_arms * self.slot_count , flattened_dist)
            
            prob_array = []
            for pair in list_of_slate_prob_pairs:
                prob_array = np.append(prob_array, pair.probability)

            # choose 2 slates in case one of them is a stub.
            [chosen_slate_index1, chosen_slate_index2] = np.random.choice(list_of_slate_prob_pairs.__len__(), size=2 , replace=False, p=prob_array)
            chosen_slate = list_of_slate_prob_pairs[chosen_slate_index1].indicator
            
            if type(chosen_slate) == str and chosen_slate == 'STUB':
                chosen_slate = list_of_slate_prob_pairs[chosen_slate_index2].indicator
            chosen_slate = chosen_slate.reshape(self.slot_count , -1)

            # construct the arm based on this chosen_slate
            chosen_arm_indices = []
            
            for l in range(self.slot_count):
                assert len(np.nonzero(chosen_slate[l])[0]) == 1
                chosen_arm_indices.append(np.nonzero(chosen_slate[l])[0][0] - l*self.item_count)
            chosen_arm = np.hstack([self.arm_set[j][chosen_arm_indices[j]] for j in range(self.slot_count)])
            actual_reward , expected_regret , best_arm = self.interact(chosen_arm)
            self.regret_arr.append(expected_regret)


            # construct the loss matrix
            loss_matrix = np.zeros(self.one_matrix_for_slots.shape)
            for l in range(self.slot_count):
                
                # construct the padded arm and get the loss for that particular arm
                padded_arm = [0 for _ in range(len(chosen_arm))]
                dim = len(padded_arm)//self.slot_count
                padded_arm[l*dim : (l+1)*dim] = self.arm_set[l][chosen_arm_indices[l]]
                arm_loss = -self.oracle.expected_reward_linear(padded_arm) if "linear" in self.alg_name else -self.oracle.expected_reward(padded_arm)
                
                # add this loss to the loss_matrix
                loss_matrix[l][chosen_arm_indices[l] + l*self.item_count] = arm_loss / (self.slot_count * intermediate_dist[l][chosen_arm_indices[l] + l*self.item_count])
            
            np.clip(loss_matrix , -1 , 1)
            # send the loss_matrix back to the mw engine
            self.mw_engine.computeAfterLossOrdered(loss_matrix)