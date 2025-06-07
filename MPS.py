import numpy as np
from utils import sigmoid

class MPS:

    def __init__(self , params):
        self.slot_count = params["slot_count"]
        self.item_count = params["item_count"]
        self.arm_dim = params["arm_dim"]
        self.alphas = [[1 for i in range(self.item_count)] for j in range(self.slot_count)]
        self.betas = [[1 for i in range(self.item_count)] for j in range(self.slot_count)]
        self.picked_embedding_indices = None

    def update_parameters(self , embedding_indices , reward):
        '''
        Update the parameters of the algorithm
        '''
        for slot_idx , embedding_index in enumerate(embedding_indices):
            self.alphas[slot_idx][embedding_index] += reward
            self.betas[slot_idx][embedding_index] += (1 - reward)
    
    def pull(self , arm_set):
        '''
        pulls the arm by maximizing over the drawn samples
        '''
        picked_embedding_indices = self.find_slotwise_argmax(arm_set)  
        return picked_embedding_indices 

    def draw_samples(self , slot_idx , arm_num):
        '''
        Draw samples from a beta distribution
        '''
        alpha = self.alphas[slot_idx][arm_num]
        beta = self.betas[slot_idx][arm_num]
        return np.random.beta(alpha , beta)
    
    def find_slotwise_argmax(self , arm_set):
        '''
        finds the arm to play by drawing samples
        '''
        picked_embedding_indices = []
        for slot_idx , slot in enumerate(arm_set):
            values = [self.draw_samples(slot_idx , arm_num) for arm_num , arm in enumerate(slot)]
            picked_arm_index = np.random.randint(0, self.item_count) if len(set(values)) == 1 else np.argsort(values)[-1]
            picked_embedding_indices.append(picked_arm_index)
        return picked_embedding_indices