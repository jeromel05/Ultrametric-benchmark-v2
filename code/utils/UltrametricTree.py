import numpy as np

class SynthUltrametricTree():
    def __init__(self, max_depth=12, p_flip=0.01, leaf_length=1000, shuffle_labels=True, noise_level=1):
        self.leaves = []
        self.max_depth = max_depth
        self.p_flip = p_flip
        self.leaf_length = leaf_length
        self.ancestor_seq = np.random.randint(0, 2, self.leaf_length, dtype=bool)
        self.shuffle_labels = shuffle_labels
        self.noise_level = noise_level
        self.tree_list = [[] for i in range(self.max_depth+1)]
        self.create_tree(self.ancestor_seq, 0, self.tree_list)
        
        self.nb_classes = len(self.leaves)
        labels = np.arange(0, self.nb_classes, 1)
        self.leaves = np.array([self.compute_noisy_seq(el) for el in self.leaves]).reshape(2**self.max_depth*self.noise_level,
                                                                                           self.leaf_length)

        if self.shuffle_labels:
            np.random.shuffle(labels)
        #self.dataset = {i: el for i, el in enumerate(grouper(self.leaves, group_size))} #incomplete='strict'))}
        self.labels = np.array([[el]*self.noise_level for el in labels]).flatten() 
        
    def bit_flip(self, ancestor_seq):
        random_seq = np.random.rand(len(self.ancestor_seq))
        return [not el_res if el_rand < self.p_flip else el_res 
                           for el_res, el_rand in zip(self.ancestor_seq, random_seq)]

    def create_tree(self, ancestor_seq, curr_depth, tree_list):
        current_depth = curr_depth + 1
        left_child = self.bit_flip(ancestor_seq)
        right_child = self.bit_flip(ancestor_seq)
        if current_depth == self.max_depth:
            self.leaves.append(left_child)
            self.leaves.append(right_child)
            tree_list[curr_depth].append([left_child, right_child])
        else:
            return tree_list[curr_depth].append([self.create_tree(left_child, current_depth, tree_list), 
                    self.create_tree(right_child, current_depth, tree_list)])
        
    def compute_noisy_seq(self, seq):
        res_seqs = [seq]
        for i in range(self.noise_level-1):
            res = seq.copy()
            rand_id = np.random.randint(0, self.leaf_length)
            res[rand_id] = not res[rand_id]
            res_seqs.append(res)
        return res_seqs
    
        

        
        
        

        
        