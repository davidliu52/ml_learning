# -*- coding: utf-8 -*-
"""
Created on Wed May 24 17:07:06 2023

utilities for the decision tree model

@author: merte
"""

import numpy as np
from matplotlib import pyplot as plt


class Node:

    def __init__(self,
                 split_dim:int=None, split_val:float=None,
                 child_left=None, child_right=None,
                 prediction:int=None):
        """Initialize Node class
        
        If a split condition and child nodes are provided: returns a parent
        node. Otherwise: returns a leaf node making the <prediction>
        
        """

        # splitting condition
        self.split_dim = split_dim           # the index of the feature dimension
        self.split_val = split_val          # the value along which to split x <= value

        # child nodes to the current node
        self.child_left = child_left        # the left child node
        self.child_right = child_right      # the right child node

        # prediction (if leaf) value
        # the output value \hat{y} of current node
        self.prediction = prediction

    def is_leaf(self):
        """ returns boolean to indicate leaf node"""
        return self.prediction is not None


class DecisionTree:

    def __init__(self, max_depth: int = 5, min_samples: int = 2):
        """Initialize DecisionTree class"""

        self.root = None                # store the root node here
        self.max_depth = max_depth      # maximum depth of the tree
        self.min_samples = min_samples  # minimum number of samples per node
        
        self.class_labels = None        # list of class labels

        #  <-- this is where we would also put choices about the splitting
        # condition, i.e. Gini or entropy / log

    def is_completed(self, depth):
        """ Check tree growth stopping criteria"""

        # return True if one of the following conditions is true
        # - max depth reached
        # - minimal number of samples in a node
        # - node is pure

        if (depth >= self.max_depth) \
                or (self._curr_no_samples < self.min_samples) \
                or (self._curr_node_pure): 
            return True

        return False
    
    def _majority_vote(self, y:np.ndarray) -> int:
        """ Return majority vote
        
        for categorical data in y, i.e. the index of the most common class
        """
        
        unique_labels, bin_counts = np.unique(y, return_counts=True)
        majority_class_idx = unique_labels[np.argmax(bin_counts)]
        
        return # ??? index of the majority class label 
    

    def _entropy(self, y: np.ndarray) -> float:
        """ Compute Shannon information entropy"""
        proportions = np.bincount(y.astype(dtype=int)) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        
        return entropy


    def _information_gain(self, y_parent: np.ndarray, index_split: np.ndarray) -> float:
        """Compute informatoin gain for given split of categorical data"""

        # number of members per child node
        N = len(index_split)  # overall number of data points
        N_left = np.sum(index_split == 1)  # members of left child
        N_right = np.sum(index_split == 0)  # members of right child
        
        # compute entropy at parent node
        H_parent = self._entropy(y_parent)
        

        # information gain will be zero if a child has no members (special case)
        if N_left == 0 or N_right == 0:
            return 0

        # compute information gain
        H_left = self._entropy(y_parent[index_split])
        H_right = self._entropy(y_parent[index_split == 0])

        return H_parent - ((N_left / N) * H_left + (N_right / N) * H_right)


    def _create_split(self, X, split_dim, split_val):
        """Split data set accordingto split condition"""

        return X[:, split_dim] <= split_val


    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """Find the best split w.r.t. information gain"""

        split = {'score': 0, 'dim': None, 'thresh': None}

        for _split_dim in range(X.shape[1]):
            X_feat = X[:, _split_dim]

            # find all possible splits along this feature dimension
            possible_splits = np.unique(X_feat)
            for _split_val in possible_splits:

                # create split
                idx_split = self._create_split(
                    X=X, split_dim=_split_dim, split_val=_split_val)

                # compute information gain
                score = self._information_gain(y, idx_split)

                # update if score was better than before
                if score > split['score']:
                    split['score'] = score
                    split['dim'] = _split_dim
                    split['thresh'] = _split_val
        
        print(f'best split: feat. dim={split["dim"]}, feat. value={split["thresh"]:.3f}, info. gain={ split["score"]:.3f}')

        return split['dim'], split['thresh']
    
    def _grow_tree(self, X:np.ndarray, y:np.ndarray, curr_depth:int=0):
        """ Build tree for given training data set
        
        This is a recursive function that calls itself over and over until it
        hits the stopping criteria. This method creates Nodes and child nodes
        
        X: [N*, n*] data setof N* samples with n* feature dimensions
        y: [N*] labels
        
        Note that (*) the number of samples differs from node to node, hence
        N* < N is the general case, where N is the overall number of samples
        in the training data set.
        """
        
        self._curr_no_samples = X.shape[0]  # current number of samples
        
        # check for purity in the currentset of labels y. True if pure
        self._curr_node_pure = # ??? true if pure, false if impure
        
        # 0.0 base condition (stopping criteria) for recursive call. return 
        # leaf node if base condition is met
        if # ??? implement base condition:
            # return a leaf node carrying the majority vote for the targets y
            return Node(prediction=self._majority_vote(y))
            
        # 0.1 if base condition is not met: recursively grow the tree:
        # 1. find best split for current data (X, y)
        split_dim, split_val = #???
        print(f'depth={curr_depth}; split condition: x_{split_dim} <= {split_val:.3f}\n')
        
        # 2.split the current data according to best split and return index
        # for data sent to left and right child node
        left_idx = self._create_split(X, split_dim, split_val)
        right_idx = ~ left_idx
        
        # 3. create child nodes and assign data using recursive call
        child_left = # ???
        child_right = # ??? 
        
        # create a node with child nodes for the given spit condition
        node =  Node(split_dim=split_dim, 
                     split_val=split_val, 
                     child_left=child_left, 
                     child_right=child_right, 
                     )
                
        return node
    
    def _traverse_tree(self, x:np.ndarray, node:Node=None):
        """ Feed data trough given tree structure. 
        
        
        This is a recursive function that feeds the incoming data sample x
        through the tree until a leaf node is met, for which the prediction
        value is returned. 
        
        x: [1, n] individual data sample
        
        returns scalar, i.e. prdicted class label
        
        """
        
        # base condition for breaking the recursive call
        # if the node is a leaf node: return majority vote prediction
        if node.is_leaf():
            return # ???
        
        # if node is not leaf, further traverse the tree
        if x[node.split_dim] <= node.split_val:
            return # ???
        else:
            return # ???
              
        
        

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit tree to training data.
        
        Generate root node and then build the tree until hitting the stopping
        criterion.
        
        """
        
        # data properties: size, dimensionality, categories
        self.n_samples, self.n_features = X.shape 
        self.class_labels = np.unique(y)  # list of all class labels 
        
        # grow the tree and append nodes to root node
        self.root = #???
        

    def predict(self, X: np.ndarray):
        """Make a prediction for unseen data. 
        
        Traverse the tree with some new data and return the prediction.
        
        """
        # feed each data sample in Xthrough the tree
        predictions = # ???
        
        return np.array(predictions)
        

        return


if __name__ == '__main__':
    
    """
    Simple 1-dimensional data set for de-bugging your code. 
    We should split at -0.5, and use a DT of depth=1
    """
    # X_train = np.expand_dims(np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]), axis=-1)
    # y_train = np.array([0, 0, 0, 1, 1, 1])
    
    # # validation data
    # X_val = np.expand_dims(np.array([-1.5, 1.8]), axis=-1)
    # y_val = np.array([0, 1])
    

    """
    Exercise data set (same as in DBSCAN lecture)
    (everything within Manhatten-Norm<=1 should be class 1)
    """
    # data set for the exercise (same as displayed in lecture on DBSCAN)
    data = np.loadtxt('decision_tree_dataset.txt', delimiter=',')
    X_train = data[:,:2]              # features
    y_train = data[:,-1].astype(int)  # targets. <int> conversion for np.bincount
    
    
    """
    Fit decision tree to training data 
    
    Feel free to eperiment with the hyperparameters max_depth and min_samples!
    """
    # create DT class object
    DT = DecisionTree()  # potentially constrain depth or min_samples

    # fit DT to data set
    DT.fit(X=X_train, y=y_train)
    
    
    """
    Investigate the DT after training:
        - how well does it perform on the training data set
        - how well does it generalize (new, unseen data)
        - illustrate the decision boundaries
    """
    # make prediction (first on training data set)   
    X_val = X_train
    y_val = y_train
    y_pred = DT.predict(X_val)
    
    print(f'\n\nground truth labels: \t {y_val}')
    print(f'predicted labels: \t \t {y_pred}')
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(X_val[y_val==0,0], X_val[y_val==0,1], linestyle='none', marker='.', markersize=10, color='blue')
    plt.plot(X_val[y_val==1,0], X_val[y_val==1,1], linestyle='none', marker='*', markersize=10, color='red')
    plt.legend(['class 0', 'class 1'])
    plt.title('ground truth')
    plt.xlabel('dim 1'); plt.ylabel('dim 2');
    
    plt.subplot(1,2,2)
    plt.plot(X_val[y_pred==0,0], X_val[y_pred==0,1], linestyle='none', marker='o', markersize=5, color='blue')
    plt.plot(X_val[y_pred==1,0], X_val[y_pred==1,1], linestyle='none', marker='+', markersize=15, color='red')
    plt.legend(['class 0', 'class 1'])
    plt.xlabel('dim 1'); plt.ylabel('dim 2');
    plt.title('prediction')
    plt.tight_layout()
    plt.show()
    
    
    # generate some unseen data
    # (everything within Manhatten-Norm<=1 should be class 1)
    X_val2 = np.array([[0, 0.8], [0, 1.2], [2, 0]])
    y_val2 = np.array([1, 0, 0])
    y_pred2 = DT.predict(X_val2)
    
    
    X_val = np.concatenate((X_val, X_val2))
    y_val = np.concatenate((y_val, y_val2))
    y_pred = np.concatenate((y_pred, y_pred2))
    

    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(X_val[y_val==0,0], X_val[y_val==0,1], linestyle='none', marker='.', markersize=10, color='blue')
    plt.plot(X_val[y_val==1,0], X_val[y_val==1,1], linestyle='none', marker='*', markersize=10, color='red')
    plt.legend(['class 0', 'class 1'])
    plt.title('ground truth')
    plt.xlabel('dim 1'); plt.ylabel('dim 2');
    
    plt.subplot(1,2,2)
    plt.plot(X_val[y_pred==0,0], X_val[y_pred==0,1], linestyle='none', marker='o', markersize=10, color='blue')
    plt.plot(X_val[y_pred==1,0], X_val[y_pred==1,1], linestyle='none', marker='+', markersize=10, color='red')
    plt.legend(['class 0', 'class 1'])
    plt.xlabel('dim 1'); plt.ylabel('dim 2');
    plt.title('prediction')
    plt.tight_layout()
    plt.show()
    
    """
    Visualize the decision boundaries by quering the DT for a grid of points
    """
    # generate some grid-like data to visualize the decision boundaries learned
    x1_grid = np.arange(-3, 3, 0.1)
    x2_grid = x1_grid
    X_grid = []
    for x1 in x1_grid:
        for x2 in x2_grid:
            X_grid.append([x1, x2])
    X_grid = np.array(X_grid)
    
    # make predictions for all grid data points
    y_pred_grid = DT.predict(X_grid)
    
    plt.figure()
    plt.plot(X_grid[y_pred_grid==0,0], X_grid[y_pred_grid==0,1], linestyle='none', marker='.', markersize=10, color='blue')
    plt.plot(X_grid[y_pred_grid==1,0], X_grid[y_pred_grid==1,1], linestyle='none', marker='*', markersize=10, color='red')
    plt.legend(['class 0', 'class 1'])
    plt.title('decision boundaries learned by DT')
    plt.xlabel('dim 1'); plt.ylabel('dim 2');
    plt.show()
    
