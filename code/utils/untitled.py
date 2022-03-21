Epaves

class Node():
        def __init__(self, seq, parent_node=None, child_nodes=None):
            self.parent_node = parent_node
            self.child_nodes = child_nodes
            self.seq = seq

        def add_children(self, child_nodes):
            self.child_nodes = child_nodes

        def is_root(self):
            return self.parent_node == None

        def is_leaf(self):
            return self.child_nodes == None

        def get_parent(self):
            return self.parent_node

        def get_children(self):
            return self.child_nodes
        
        def get_seq(self):
            return self.seq
        
class SynthTreeV2():
    def __init__(self, max_depth=12, p_flip=0.1, leaf_length=1000, label_level=0, shuffle_labels=True):
        self.max_depth = max_depth
        self.p_flip = p_flip
        self.leaf_length = leaf_length
        self.label_level = label_level
        self.root_node = Node(np.random.randint(0, 2, self.leaf_length, dtype=bool))
        self.shuffle_labels = shuffle_labels
        
        self.node_list = [self.root_node]
        self.tree = self.create_tree(self.root_node, 0, self.node_list)
        self.leaves=[]
        self.get_leaves(self.root_node, self.leaves)
        group_size = self.compute_group_size()
        self.nb_classes = int(len(self.leaves) / group_size)
        labels = np.arange(0, self.nb_classes, 1)
        print(f"#samples per class: {group_size}, #samples tot: {len(self.leaves)}, #classes = {self.nb_classes}")

        if self.shuffle_labels:
            np.random.shuffle(labels)
        self.labels = np.array([[el]*group_size for el in labels]).flatten()
        
    def bit_flip(self, ancestor_seq):
        random_seq = np.random.rand(len(self.root_node.get_seq()))
        return [not el_res if el_rand < self.p_flip else el_res 
                           for el_res, el_rand in zip(self.root_node.get_seq(), random_seq)]

    def create_tree(self, parent_node, curr_depth, node_list):
        current_depth = curr_depth + 1
        left_seq = self.bit_flip(parent_node.get_seq())
        right_seq = self.bit_flip(parent_node.get_seq())
        left_child = Node(seq=left_seq, parent_node=parent_node, child_nodes=None)
        right_child = Node(seq=right_seq, parent_node=parent_node, child_nodes=None)
        parent_node.add_children([left_child, right_child])
        node_list.append(left_child)
        node_list.append(right_child)
        
        if current_depth < self.max_depth:
            self.create_tree(left_child, current_depth, node_list)
            self.create_tree(right_child, current_depth, node_list)

    def compute_group_size(self):
        if self.label_level == 0: 
            return 1
        else:
            i = np.arange(0, self.label_level+1)
            return np.sum(2**i)
    
    def get_leaves(self, root_node, res):
        if root_node.is_leaf: 
            res.append(root_node)
        else:
            for child_node in root_node.get_children:
                res.append(get_leaves(child_node, res))
                
def create_tree(self, ancestor_seq, curr_depth):
        current_depth = curr_depth + 1
        left_child = self.bit_flip(ancestor_seq)
        right_child = self.bit_flip(ancestor_seq)
        if current_depth == self.max_depth:
            self.leaves.append(left_child)
            self.leaves.append(right_child)
            return [ancestor_seq, [left_child, right_child]]       
        else:
            if current_depth == self.max_depth-self.label_level+1:
                self.leaves.append(ancestor_seq)
            if current_depth >= self.max_depth-self.label_level+1:
                self.leaves.append(left_child)
                self.leaves.append(right_child)
            return [ancestor_seq, [self.create_tree(left_child, current_depth), 
                    self.create_tree(right_child, current_depth)]]
                
#-----------------------------------------------------------------

"""    
def shuffleblocks(chain,blockl):
    if blockl==1:
        return np.random.shuffle(chain)
    else:
        nb=int(np.floor(chain.shape[0]/blockl)) # number of blocks
        ns=int(nb*10) # number of shuffles

        for i in range(0, ns):
            print(i)
            fi1=np.random.randint(0, high=nb)*blockl
            fi2=np.random.randint(0, high=nb)*blockl
            
            count=0
            while(fi2==fi1 and count<1000):
                fi2=np.random.randint(0, high=nb)*blockl
                count=count+1
                
            chain_buf=chain[fi1:(fi1+blockl)]
            chain[fi1:(fi1+blockl)]=chain[fi2:(fi2+blockl)]
            chain[fi2:(fi2+blockl)]=chain_buf
            
        return chain
"""

#-----------------------------------------------------------------------

class OneExampleSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, chain):
        self.data_source = data_source
        self.chain = chain
        
    def __iter__(self):
        return iter(chain)

    def __len__(self):
        return len(self.data_source)