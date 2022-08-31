import pandas as pd
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import *

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor


def get_rf_gb_(X,Y,n,flag): 
    # Given flag, observation and number of the tree you want to have, get the random forest or gradient boosting trees
    if flag==1:         
        rf = RandomForestRegressor(n_estimators=n,random_state=1)
        rf = rf.fit(X,Y)
    elif flag==2:
        rf = RandomForestClassifier(n_estimators=n,random_state=1)
        rf = rf.fit(X,Y) 
    elif flag==3:
        rf = GradientBoostingClassifier(n_estimators=n,random_state=1)
        rf = rf.fit(X,Y)  
    elif flag==4:
        rf = GradientBoostingRegressor(n_estimators=n,random_state=1)
        rf = rf.fit(X,Y)  
    else:
        print("flag can take values 1, 2, 3 and 4")
    return rf
      
def get_input(rf_gb,flag): 
    # This function captures the input tree ensemble and return list of trees embedded
    trees=list()
    if flag==1:# Random forests Regressor
        for i in range(rf_gb.n_estimators): 
            trees.append(rf_gb.estimators_[i])
    elif flag==2:# Random forests Classifier
        for i in range(rf_gb.n_estimators): 
            trees.append(rf_gb.estimators_[i])
    elif flag==3: # Gradient Boosting Classifier
         for i in range(rf_gb.n_estimators): 
            trees.append(rf_gb.estimators_[i][0])
    elif flag==4: # Gradient Boosting Regressor
         for i in range(rf_gb.n_estimators): 
            trees.append(rf_gb.estimators_[i][0])
    return trees 

def is_it_leaf(trees_given,t): 
    #This function returns an array of the boolean value, telling if it is leaf of the tree t
    tree=trees_given[t].tree_
    n_nodes=tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    # seed is the root node id
    stack = [0]  
    while len(stack) > 0:
        # pop() removes an element from the list (the last element by default) and returns the value of that element.
        node_id = stack.pop()
        if (children_left[node_id] != children_right[node_id]):# i.e., -1=-1 or not
            stack.append(children_left[node_id])
            stack.append(children_right[node_id])
        else:
            is_leaves[node_id] = True
    #returns an array of boolean values about whether it is a leaf or not 
    return is_leaves 

def GETLEAF(trees_given,t,x_given): 
    #This function determines the leaf to which tree t maps x
    v=0
    while is_it_leaf(trees_given,t)[v]==False:
        if x_given[V(trees_given,t,v),C(trees_given,t,v)]==1:
            v=trees_given[t].tree_.children_left[v]
        else:
            v=trees_given[t].tree_.children_right[v]
    return v

def splits(trees_given,t): 
    #return an array of splits(not leaf) of tree
    return np.arange(trees_given[t].tree_.node_count)[is_it_leaf(trees_given,t)==False] 

def leaves(trees_given,t):
    #return an array of leaf of tree
    return np.arange(trees_given[t].tree_.node_count)[is_it_leaf(trees_given,t)==True]

def total_split_variable(trees_given): 
    # This function returns the set of independent variables that are used in split conditions
    feature_set=set([])
    for tree in trees_given:
        # A value less than 0 means dummy values, i.e., a leaf node.
        feature=tree.tree_.feature[tree.tree_.feature>=0]
        # Put the features of all the trees together
        feature_set=feature_set|set(feature) 
    # It should be the variables used in the split point, which means that not all features will be used
    return feature_set

def V(trees_given,t,s): 
    #This function returns variable that participates in split s
    tree=trees_given[t].tree_
    feature=tree.feature[s]
    # V(s) in the model, where s is contained in a tree, is the variable involved in split s, and has only one value
    return feature 

def split_values(trees_given,i): 
    #This function returns array of unique split points in ascendng order
    values=np.array([])
    # If the variable is in total_split_variable, the split values of all trees for the variable are arranged from small to large
    if i in total_split_variable(trees_given): 
        for tree in trees_given:
            # If a leaf is required, the return value is -2
            feature=set(tree.tree_.feature[tree.tree_.feature>=0])
            if i in feature:
                # Write the split values of variable in each tree into the array values
                values=np.append(values,tree.tree_.threshold[tree.tree_.feature==i])
    #in ascendng order
    values=np.unique(np.sort(values))
    # returns array of unique split points in ascendng order
    return values

def C(trees_given,t,s): 
    #Set of values of variables i that participate in split
    #The expression in the paper is not right, since there is only one threshold in each split (This means that the sum of the two constraints in the model with respect to C(s) is incorrect because there is only one value in C. And no summation notation is required)
    # Determine the split point s of tree T
    threshold=trees_given[t].tree_.threshold[s]
    #V(s) in the model, where s is contained in a tree, is the number of the variable involved in split s, and has only one value
    feature=V(trees_given,t,s)
    # np.where(condition) When there is only one argument, that argument represents the condition. When the condition is true, np.where(condition) returns the coordinates of each element that meets the condition, in the form of a tuple
    return int(np.where(split_values(trees_given,feature)==threshold)[0])

def K(trees_given,i): 
    #K_i is the number of unique split points for variable i
    return split_values(trees_given,i).shape[0]


def prediction(trees_given,t,l,flag): 
    #prediction of tree t, leaf l
    tree=trees_given[t].tree_
    #This is a classification tree and return which class the leaf predicts
    if flag==2: 
        return np.argmax(tree.value[l,0,:])
    elif flag==3:
        # argmax function returns the index with the most value, which is the sequence number of the category varible
        return np.argmax(tree.value[l,0,:]) 
    else: 
        #this is a regression tree 
        return tree.value[:,0,0][l] 
    
def right_leaf(trees_given,t,s):  
    #return a list of all the right leaf of tree t, node s
    right_leaves=[]
    tree=trees_given[t].tree_ 
    n_nodes=tree.node_count 
    # All left children nodes, -1 means no children (leaves)
    children_left = tree.children_left 
    # All right children nodes, -1 means no children (leaves)
    children_right = tree.children_right 
    stack = [s] 
    node_id = stack.pop()
    # If the left and right nodes of these nodes are not the same,
    if (children_left[node_id] != children_right[node_id]):
        # Add the right child node
        stack.append(children_right[node_id]) 
        # Same returns all right_leaves. Because it is a leaf, there is no right of S.
    else:
        return right_leaves 
    while len(stack) > 0: 
        node_id = stack.pop()
        # it means it is not a leaf
        if (children_left[node_id] != children_right[node_id]):
            # Add both left and right nodes
            stack.append(children_left[node_id])
            stack.append(children_right[node_id])
        else:
            # Add right nodes
            right_leaves.append(node_id)
    return right_leaves 

def as_right_leaf(trees_given,t,l): 
    #return an array of splits whose right leaf is l
    split=np.array([])
    tree=trees_given[t].tree_
    n_nodes=tree.node_count
    for i in range(n_nodes): 
        # Find the right leaf of node i and determine if l is in it or not
        if l in set(right_leaf(trees_given,t,i)):
            # add it in the split array
            split=np.append(split,np.array([i])) 
    return split 

def left_leaf(trees_given,t,s): 
    #return a list of all the left leaf of tree t, node s
    left_leaves=[]
    tree=trees_given[t].tree_
    n_nodes=tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    stack = [s]
    node_id = stack.pop()
    if (children_left[node_id] != children_right[node_id]):
        stack.append(children_left[node_id])
    else:
        return left_leaves
    while len(stack) > 0:
        node_id = stack.pop()
        if (children_left[node_id] != children_right[node_id]):
            stack.append(children_left[node_id])
            stack.append(children_right[node_id])
        else:
            left_leaves.append(node_id)
    return left_leaves

def as_left_leaf(trees_given,t,l): 
    #return an array of splits whose left leaf is l
    split=np.array([])
    tree=trees_given[t].tree_
    n_nodes=tree.node_count
    for i in range(n_nodes):
        if l in set(left_leaf(trees_given,t,i)):
            split=np.append(split,np.array([i])) 
    return split

def x_to_one(trees_given,x_given): 
    #given an observation of X, x_V(s),j = 1 if j belongs to C(s)
    x_one={}
    for t in range(len(trees_given)):
        for s in splits(trees_given,t):
            # C(trees_given,t,s): the number of the S split point of the t tree.
            # split_values(trees_given,V(trees_given,t,s))：This is the split value at which the variable V(s) participates.
            #  x_given[V(trees_given,t,s)] : The value of a variable
            if x_given[V(trees_given,t,s)] - split_values(trees_given,V(trees_given,t,s))[C(trees_given,t,s)] <= 0:
                x_one[V(trees_given,t,s),C(trees_given,t,s)] = 1
            else:
                x_one[V(trees_given,t,s),C(trees_given,t,s)] = 0
    return x_one

def z(trees_given,t,x_given):  
    #z_t,l=1 if l = l*; otherwise, 0
    l_star=GETLEAF(trees_given,t,x_given)
    tree=trees_given[t].tree_
    node=tree.node_count
    z_return=np.zeros(node)
    z_return[l_star]=1
    return z_return

def root_partition(n,trees):
    '''
    Input:
        n (integer): The partition subsize n
        trees (list): All trees
        
   Output:
       tree_root_partition(list)
    '''
    tree_root_partition=[trees[i: i+n] for i in range(0, len(trees), n)]
    return tree_root_partition

def nonroot_partition_refinement(root_partition):
    '''
    The function will refine the partition list.
    
    Input:
        Tree_root_partition(list): The root partition list we want to refine
        
   Output:
       nonroot_node_partition(list)
    '''
    
    i=1
    tree_root_partition=root_partition
    nonroot_node_partition=list()
    
    # we have a time limit for refinement.
    start_time = time.time() 
    time_limit=120
    end_time=time.time()+time_limit
    
    # The elements are merged pairwise
    while i<=int(math.floor(len(tree_root_partition)/2)):
        q=tree_root_partition[2*i-2]+tree_root_partition[2*i-1]
        nonroot_node_partition.append(q)
        i=i+1
        if time.time()>=end_time:
            break
            
    # Combine the two lists
    del tree_root_partition[0:2*i-2]
    nonroot_node_partition=nonroot_node_partition+tree_root_partition
    
    return nonroot_node_partition

