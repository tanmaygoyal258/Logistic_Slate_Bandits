import numpy as np
from optimization import fit_online_logistic_estimate, fit_online_logistic_estimate_bar
from utils import sigmoid, dsigmoid, weighted_norm, gaussian_sample_ellipsoid , probit , dprobit , regularized_log_loss
from datetime import datetime
from scipy.optimize import minimize


class Slate_GLM_TS():
    def __init__(self, params , kappa = None , warmup = False):
        self.reward_type = params["reward_type"]
        self.reward_func = sigmoid if self.reward_type == "logistic" else probit
        self.d_reward_func = dsigmoid if self.reward_type == "logistic" else dprobit
        
        self.item_count = params["item_count"]
        self.slot_count = params["slot_count"]
        self.horizon = params["horizon"]
        self.arm_dim = params["arm_dim"]
        self.l2reg = 5
        self.failure_level = params["failure_level"]
        self.param_norm_ub = params["param_norm_ub"]
        self.dim = self.slot_count * self.arm_dim

        self.warmup_flag = warmup
        if self.warmup_flag:
            self.kappa = kappa
            self.warmup_length = self.param_norm_ub**6 * self.kappa * self.dim**2 * np.log(self.horizon/self.failure_level)**2
            if self.warmup_length > self.horizon / 10:
                self.warmup_length = self.horizon / 10
            print(f"Warmup length is {self.warmup_length}")
            self.warmup_arms = []
            self.warmup_rewards = []

        self.vtilde_matrix = self.l2reg * np.eye(self.dim)
        self.vtilde_matrix_inv = (1 / self.l2reg) * np.eye(self.dim)
        self.v_matrices_inv = [(1 / self.l2reg) * np.eye(self.arm_dim) for _ in range(self.slot_count)]
        self.v_matrices = [self.l2reg * np.eye(self.arm_dim) for _ in range(self.slot_count)]        
        self.theta = np.zeros((self.dim,))
        self.theta_tilde = [np.zeros((self.arm_dim,)) for _ in range(self.slot_count)]
        
        self.conf_radius = 0
        self.cum_loss = 0
        self.ctr = 1

        self.eigenvalue_flag = "eigenvalues" in list(params.keys())
        if self.eigenvalue_flag:
            self.eigenvalues = [[] for _ in range(self.slot_count)] 

    def update_parameters(self, arm , reward):
        if not self.warmup_flag:
            
            # compute new estimate theta
            self.theta = np.real_if_close(fit_online_logistic_estimate(arm=arm,
                                                                    reward=reward,
                                                                    current_estimate=self.theta,
                                                                    vtilde_matrix=self.vtilde_matrix,
                                                                    vtilde_inv_matrix=self.vtilde_matrix_inv,
                                                                    constraint_set_radius=self.param_norm_ub,
                                                                    diameter=self.param_norm_ub,
                                                                    precision=1/self.ctr))
            # compute theta_bar (needed for data-dependent conf. width)
            theta_bar = np.real_if_close(fit_online_logistic_estimate_bar(arm=arm,
                                                                        current_estimate=self.theta,
                                                                        vtilde_matrix=self.vtilde_matrix,
                                                                        vtilde_inv_matrix=self.vtilde_matrix_inv,
                                                                        constraint_set_radius=self.param_norm_ub,
                                                                        diameter=self.param_norm_ub,
                                                                        precision=1/self.ctr))
            disc_norm = np.clip(weighted_norm(self.theta-theta_bar, self.vtilde_matrix), 0, np.inf)

            # sensitivity check
            sensitivity_bar = self.d_reward_func(np.dot(theta_bar, arm))
            sensitivity = self.d_reward_func(np.dot(self.theta, arm))        
            if sensitivity_bar / sensitivity > 2:
                msg = f"\033[95m Oops. ECOLog has a problem: the data-dependent condition was not met. This is rare; try increasing the regularization (self.l2reg) \033[95m"
                raise ValueError(msg)
            
            # update sum of losses and ctr
            coeff_theta = self.reward_func(np.dot(self.theta, arm))
            loss_theta = -reward * np.log(coeff_theta) - (1-reward) * np.log(1-coeff_theta)
            coeff_bar = self.reward_func(np.dot(theta_bar, arm))
            loss_theta_bar = -reward * np.log(coeff_bar) - (1-reward) * np.log(1-coeff_bar)
            self.cum_loss += 2*(1+self.param_norm_ub)*(loss_theta_bar - loss_theta) - 0.5*disc_norm
        

        # update matrices
        sensitivity = self.d_reward_func(np.dot(self.theta, arm)) if not self.warmup_flag else 1.0/self.kappa
        self.vtilde_matrix += sensitivity * np.outer(arm, arm)
        self.vtilde_matrix_inv += - sensitivity * np.dot(self.vtilde_matrix_inv,
                                                        np.dot(np.outer(arm, arm), self.vtilde_matrix_inv)) / (
                                          1 + sensitivity * np.dot(arm, np.dot(self.vtilde_matrix_inv, arm)))
        
        # store the eigenvalues if needed
        if self.eigenvalue_flag:
            for slot in range(self.slot_count):
                matrix = self.v_matrices[slot]
                eigval , _ = np.linalg.eigh(matrix)
                self.eigenvalues[slot].append(eigval[0])
        
        # extract the slotwise arm without the zeros
        slot_wise_arms = []
        for idx in range(self.slot_count):
            slot_wise_arms.append(arm[idx * self.arm_dim : (idx + 1) * self.arm_dim])

        # update the slotwise parameters
        for idx , a in enumerate(slot_wise_arms):
            self.v_matrices[idx] += sensitivity * np.outer(a, a)
            self.v_matrices_inv[idx] = self.sherman_morrison_update(self.v_matrices_inv[idx] , sensitivity**0.5 * a , sensitivity**0.5 * a)
        
        self.ctr += 1
        
        if self.warmup_flag:
            self.warmup_arms.append(arm)
            self.warmup_rewards.append(reward)

            if self.ctr > self.warmup_length:
                print("Updating parameters after warmup")
                self.warmup_flag = False
                
                self.theta = minimize(regularized_log_loss , np.zeros(self.dim) , \
                                    args = (self.warmup_arms , self.warmup_rewards , self.reward_func , self.l2reg)).x
                self.theta /= np.linalg.norm(self.theta) * self.param_norm_ub
                
                self.vtilde_matrix = self.l2reg * np.eye(self.dim)
                self.vtilde_matrix_inv = (1 / self.l2reg) * np.eye(self.dim)
                self.v_matrices_inv = [(1 / self.l2reg) * np.eye(self.arm_dim) for _ in range(self.slot_count)]
                self.v_matrices = [self.l2reg * np.eye(self.arm_dim) for _ in range(self.slot_count)]        

    def pull(self, arm_set):
        # bonus-based version (strictly equivalent to param-based for this algo) of OL2M
        if not self.warmup_flag:
            self.update_ucb_bonus()
            self.theta_tilde = [gaussian_sample_ellipsoid(self.theta[i * self.arm_dim : (i + 1) * self.arm_dim], self.v_matrices[i] , self.conf_radius) \
                            for i in range(self.slot_count)]
        picked_embedding_indices= self.find_slotwise_argmax(arm_set)  
        return picked_embedding_indices

    def update_ucb_bonus(self):
        """
        Updates the ucb bonus function (a more precise version of Thm3 in ECOLog paper, refined for the no-warm up alg)
        """
        gamma = np.sqrt(self.l2reg) / 2 + 2 * np.log(
            2 * np.sqrt(1 + self.ctr / (4 * self.l2reg)) / self.failure_level) / np.sqrt(self.l2reg)
        res_square = 2*self.l2reg*self.param_norm_ub**2 + (1+self.param_norm_ub)**2*gamma + self.cum_loss
        res_square = max(0 , res_square)
        self.conf_radius = np.sqrt(res_square)

    def compute_optimistic_reward(self, arm , slot_idx):
        """
        Returns prediction + exploration_bonus for arm.
        """
        if self.warmup_flag:
            pred_reward = weighted_norm(arm , self.v_matrices_inv[slot_idx])
        else:
            pred_reward = np.sum(self.theta_tilde[slot_idx] * arm)
        return pred_reward
    
    def sherman_morrison_update(self , v_inv , vec1 , vec2):
        '''
        implements the sherman morrison update for inverse of rank 1 additions
        '''
        return v_inv - v_inv@np.outer(vec1 , vec2)@v_inv/(1 + np.dot(vec1, v_inv@vec2))

    def find_slotwise_argmax(self, arm_set):
        """
        Returns the slotwise-arms that maximizes the optimistic reward.
        """
        picked_embedding_indices = []
        for idx , slot in enumerate(arm_set):
            values = [self.compute_optimistic_reward(arm , idx) for arm in slot]
            picked_arm_index = np.random.randint(0, self.item_count) if len(set(values)) == 1 else np.argsort(values)[-1]
            picked_embedding_indices.append(picked_arm_index)
        return picked_embedding_indices
    