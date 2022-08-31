import pandas as pd
import numpy as np
from prepareziyue import *
from gurobipy import *
from copy import deepcopy
from Node import *

# Strong branching
def Strong_branching(S,B,f):
    '''
    The function get the strong branching node. Please check more details of this in algorithm 3 in the article.
    
    Input:
        S(Node): The node
        B(list): Branch ordering
        f(float): objective value of a feasible solution
        
    Output:
        S(Node): The node we will branch
        B[temp[0]](tuple): The varible i and split value
        temp[0](integer): The number of split node in Branching ordering
        False or True: try other split nodes in branching ordering or not
    
    '''
    # record the split node number
    i=0
    temp=[]
    
    # try to find the split node as strong branching.
    while i <len(B):
        print("S node is ",S.get_LB(),S.get_UB())
        # some split nodes can not be used as branching nodes because they are outside the domain.
        while B[i][1]>=S.get_UB()[B[i][0]] or B[i][1]<=S.get_LB()[B[i][0]]:
            i=i+1
            print("The number of the node is ",i)
            if i>= len(B):
                return S,B[0],i,True
            
        # try to find the split node as strong branching.   
        c1, c2 = get_children(S, B[i][0], B[i][1])
        
        # calculate the convex part
        print("child 1 is ",c1.get_LB(),c1.get_UB())
        print("child 2 is ",c2.get_LB(),c2.get_UB())
        c1_convex_obj=c1.get_Convex_obj_val()
        c2_convex_obj=c2.get_Convex_obj_val()
        
        #  if the node is pruning or not
        if max(c1_convex_obj,c2_convex_obj)>f:
            #  if the node will is pruning, use it as strong branching node
            if c1_convex_obj<c2_convex_obj:
                print("We find the strong branching node.")
                return c1,B[i],i,True
            #  if the node will is pruning, use it as strong branching node
            else:
                print("We find the strong branching node.")
                return c2,B[i],i,True
        else:
            temp.append(i)
            
        i=i+1
        print("The number of the node is ",i)
        # If we can not find the strong branching within B,
        if i>= len(B):
            #  we return the split node with the most weight
            if len(temp)>0:
                print("We did not find the strong branching node.")
                return S,B[temp[0]],temp[0],False
            # Or we try other split nodes, because no split nodes can be used in B
            else:
                print("We did not find the branching node. We will try it again.")
                return S,B[0],i,True
            
    #  we return the split node with the most weight
    print("We did not find the strong branching node.")
    return S,B[temp[0]],temp[0],False


# weight calculation
def weight_calculation(trees):
    '''
    The function calculates the weights of the split nodes. Please find more details of this in algorithm 3 in the article.
    
    Input:
        Trees(list): The list of trees
        
    Output:
        Branch_ordering(list): The list of branch variable i and split value in descending order. 
    
    '''
    weight={}
    weight_2={}
    
    #calculate the weights
    for i in total_split_variable(trees):
        V=split_values(trees,i)
        for v in V:
            for t in range(len(trees)):
                w=0
                if i in total_split_variable([trees[t]]):
                    for s in splits([trees[t]], 0):
                        if abs(v-trees[t].tree_.threshold[s])<1e-5:
                            w=w+(1/trees[t].tree_.node_count*(len(right_leaf(trees,t,s))+len(left_leaf(trees,t,s))))
                    weight[i,v,t]=w
            weight_2[i,v]=sum(weight[i,v,t] for t in range(len(trees)))
            
    # sort it in descending order using weights
    B=deepcopy(weight_2)
    z=sorted(zip(B.values(),B.keys()))
    z.reverse()
    z_1=list(z)
    
    # we get the list of branch variable i and split value and remove weights.
    Branch_ordering=[z_1[i][1] for i in range(len(z_1))]
    return Branch_ordering

def B_B(S_given, Branch_ordering,choice):
    
    '''
    The function uses the B&B Algorithm. Please find more details of this in algorithm 1 in the article.
    
    Input:
        S(Node): The list of trees
        Branch_ordering(list): The list of split nodes.
        
    Output: None      
    
    '''    
    S=S_given
    begin_time=time.time()
    print("Check begin time:",time.ctime(begin_time))
    #b_convex = S.get_Convex_obj_val()
    #b_GBT = S.get_GBT_obj_val()
    #f=S.get_feasible_sol()
    f=S.get_feasible_sol()
    Q= {S}
    begin=0
    B=Branch_ordering
    i=0
    b_best_lower_bound=100000000
    b_lower_bounds={}
    b_upper_bounds={}
    a=True
    b=0
    
    while len(Q) > 0:
        # Step 1-5 in algorithm 1
        S = Q.pop()
        f=min(f,S.get_feasible_sol())
        b_convex = S.get_Convex_obj_val()
        print("convex part of S: ",b_convex)
        b_GBT = S.get_GBT_obj_val()
        print("GBT part of S: ",b_GBT)
    
        #check some information of algorithm 1 in the article
        b_best_lower_bound=min(b_best_lower_bound,b_convex+b_GBT)
        b_upper_bounds[time.time()]={f}
        b_lower_bounds[time.time()]={b_best_lower_bound}
        print("Check step 1-6 time: ",time.ctime(time.time()),"best_lower_bound: ",b_best_lower_bound,"feasible solution: ",f)
    
        # We do not use the strong branching here. We just use the weights of each split node in descend order and pick one by one.
        if abs(f-b_GBT-b_convex)>1e-5:
                if choice==0:
                    S_1=S
                    b_convex=b_convex
                    b_GBT=b_GBT
                    split=B.pop(i)
                    while split[1]>=S.get_UB()[split[0]] or split[1]<=S.get_LB()[split[0]]:
                        split=B.pop(i+1)
                    print("Split node: ", split)
                
        # We use the strong branching here.        
                elif choice==1:
                     if abs(f-b_GBT-b_convex)>1e-5:
                        S_1=S
                        b_convex=b_convex
                        b_GBT=b_GBT
                        while a:
                            b=b+begin
                            S_1,split,begin,a=Strong_branching(S_1,B[b:b+5],f)
                            print("Here the split node is",begin,b)
                        print("Split node: ", split)
            
        
                # nonroot partition refinement
                P_=nonroot_partition_refinement(S_1.get_tree_partitionl())
        
                # update the tree partition of the node S
                S_1.update_tree_partition(P_)
                b_convex = S_1.get_Convex_obj_val()
                b_GBT=S_1.get_GBT_obj_val()
                print("Check the GBT lower bound after the partition refinement:",b_GBT)
        
                # get the children of node S
                S_left,S_right=get_children(S_1, split[0], split[1])
        
                #check some information of algorithm 1 in the article
                b_best_lower_bound=max(b_best_lower_bound,b_convex+b_GBT)
                b_lower_bounds[time.time()]={b_best_lower_bound}
                b_upper_bounds[time.time()]={f}
                print("Check time of step 6-18:",time.ctime(time.time()),"best_lower_bound: ",b_best_lower_bound,"feasible solution: ",f)
        
        
                for S_child in {S_left,S_right}:
                    print("check time of step 19",time.ctime(time.time()))
            
                    # Get the the convex part objective value and GBT part objective value
                    a=S_child.get_Convex_obj_val()
                    b=S_child.get_GBT_obj_val()
                    print("child node has the convex part objective value and GBT part objective value ",a,b,"and the global value ",a+b)
                    print("the feasible solution ",f)
                    if a+b<f and math.isclose(a+b,f)==False:
                        # add child into Q
                        Q=Q|{S_child}
                        print("check step 20: we added a child node! ",time.ctime(time.time()))
                
                print("check step 23: ",time.ctime(time.time()),"best_lower_bound: ",b_best_lower_bound,"feasible solution: ",f)
                if choice==0:
                    i=i+1
                    split=B.pop(i)
        
        if time.time()-begin_time>=20*60:
            break
            
    return b_lower_bounds,b_upper_bounds,begin_time
