import numpy as np
from optimization import fit_online_logistic_estimate, fit_online_logistic_estimate_bar
from utils import sigmoid, dsigmoid, weighted_norm, probit , dprobit 


class Slate_GLM_OFU():
    def __init__(self, params):

        self.reward_type = params["reward_type"]
        self.reward_func = sigmoid if self.reward_type == "logistic" else probit
        self.d_reward_func = dsigmoid if self.reward_type == "logistic" else dprobit

        self.item_count = params["item_count"]
        self.slot_count = params["slot_count"]
        self.horizon = params["horizon"]
        self.arm_dim = params["arm_dim"]
        self.dim = self.slot_count * self.arm_dim

        self.l2reg = 5
        self.failure_level = params["failure_level"]
        self.param_norm_ub = params["param_norm_ub"]
        
        self.vtilde_matrix = self.l2reg * np.eye(self.dim)
        self.vtilde_matrix_inv = (1 / self.l2reg) * np.eye(self.dim)
        self.v_matrices_inv = [(1 / self.l2reg) * np.eye(self.arm_dim) for _ in range(self.slot_count)]
        self.v_matrices = [self.l2reg * np.eye(self.arm_dim) for _ in range(self.slot_count)]        
        self.theta = np.zeros((self.dim,))
        
        self.conf_radius = 0
        self.cum_loss = 0
        self.ctr = 1

        self.eigenvalue_flag = "eigenvalues" in list(params.keys())
        if self.eigenvalue_flag:
            self.eigenvalues = [[] for _ in range(self.slot_count)] 

    def update_parameters(self, arm , reward):
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

        # update matrices
        sensitivity = self.d_reward_func(np.dot(self.theta, arm))
        self.vtilde_matrix += sensitivity * np.outer(arm, arm)
        self.vtilde_matrix_inv += - sensitivity * np.dot(self.vtilde_matrix_inv,
                                                        np.dot(np.outer(arm, arm), self.vtilde_matrix_inv)) / (
                                          1 + sensitivity * np.dot(arm, np.dot(self.vtilde_matrix_inv, arm)))

        # sensitivity check
        sensitivity_bar = self.d_reward_func(np.dot(theta_bar, arm))
        if sensitivity_bar / sensitivity > 2:
            msg = f"\033[95m Oops. ECOLog has a problem: the data-dependent condition was not met. This is rare; try increasing the regularization (self.l2reg) \033[95m"
            raise ValueError(msg)

        # update sum of losses and ctr
        coeff_theta = self.reward_func(np.dot(self.theta, arm))
        loss_theta = -reward * np.log(coeff_theta) - (1-reward) * np.log(1-coeff_theta)
        coeff_bar = self.reward_func(np.dot(theta_bar, arm))
        loss_theta_bar = -reward * np.log(coeff_bar) - (1-reward) * np.log(1-coeff_bar)
        self.cum_loss += 2*(1+self.param_norm_ub)*(loss_theta_bar - loss_theta) - 0.5*disc_norm
        self.ctr += 1

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
        for idx , arm in enumerate(slot_wise_arms):
            self.v_matrices[idx] += sensitivity * np.outer(arm, arm)
            self.v_matrices_inv[idx] = self.sherman_morrison_update(self.v_matrices_inv[idx] , sensitivity**0.5 * arm , sensitivity**0.5 * arm)

    def pull(self, arm_set):
        # bonus-based version (strictly equivalent to param-based for this algo) of OL2M
        self.update_ucb_bonus()
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
        norm = weighted_norm(arm, self.v_matrices_inv[slot_idx])
        local_theta = self.theta[slot_idx * self.arm_dim : (slot_idx + 1) * self.arm_dim]
        pred_reward = np.sum(local_theta * arm)
        bonus = self.conf_radius * norm
        return pred_reward + bonus
    
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