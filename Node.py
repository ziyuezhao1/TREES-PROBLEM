import numpy as np
from copy import deepcopy

from prepareziyue import *
from BB import *
from gurobipy import *
from ziyuesolver_2 import *

class Node(object):
    
    '''
        Node is a class used in branch and bound algorithm. 
        The node has many attributes and methods related to GBT part 
        and convex part of the tree ensemble optimization problem. Please 
        check README.md to get articles about branch and bound algorithm 
        and tree ensemble models.

        Input:
        lb (list): a lower bound vector of variable vector x
        ub (list): a upper bound vector of variable vector x
        N(integer): the root partition subset size N
        trees_given(list): The list of trees

    '''
    def __init__(self, lb, ub, N, flag, trees_given):
        self.lb = lb
        self.ub = ub
        self.N=N
        self.trees=trees_given
        self.tree_root_partition=root_partition(self.N,self.trees)
        self.flag=flag
        self.global_sol = None
        self.Convex_obj_val = None
        self.GBT_obj_val = None
        self.f=None

    # Each node has a tree partition set. And sometimes we need a nonroot tree partition refinement of the node. 
    def update_tree_partition(self,tree_partition):
        self.tree_root_partition=tree_partition
        
    # Get the tree partition of the node. 
    def get_tree_partitionl(self):
        return self.tree_root_partition
    
    # Get the lower bound of domain of the node. 
    def get_LB(self):
        return self.lb
    
    # Get the upper bound of domain of the node. 
    def get_UB(self):
        return self.ub
    
    # Get the feasible solution of the node. 
    def get_feasible_sol(self):
        self.f=feasible_solution_solver(self.trees, self.flag, self.lb,self.ub)
        return self.f
    
    # Get the GBT part objective value of the node. 
    def get_GBT_obj_val(self):
        GBT_obj=get_GBT_lower_bound(self.tree_root_partition,self.flag,self.lb, self.ub)
        self.GBT_obj_val=GBT_obj
        return self.GBT_obj_val
    
    # Get the convex part objective value of the node.   
    def get_Convex_obj_val(self):
        Convex_obj=get_convex_bound(self.trees,self.lb,self.ub)
        self.Convex_obj_val=Convex_obj
        return self.Convex_obj_val
    

def get_children(v, branch_ind, d):
    
    '''
    The function will get children of node v. For example,
       x[branch_ind] <= d, x[branch_ind] >= d
    
    Input:
        v (Node): The father node
        branch_ind (int): The braching variable i
        d(float): The braching value
        
   Output:
       left_child(Node), right_child(Node)
    '''
    
    # left child
    left_child = deepcopy(v)
    left_child.ub[branch_ind] = d
    
    # right child
    right_child = deepcopy(v)
    right_child.lb[branch_ind] = d
    
    return left_child, right_child


