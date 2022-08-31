import pandas as pd
import numpy as np
from prepareziyue import *
from gurobipy import *

def MIP_solver(trees_given, flag_given, lb, ub):     
    trees=trees_given
    flag=flag_given 
    # lamada in 10(a)
    lama=np.zeros(len(trees)) 
    for i in range(len(trees)):
        #weight =1/t
        lama[i] = 1 / len(trees) 
        
    #create initial solution for alpha, beta, gamma in 10(b)
    alpha={}
    beta={}
    gamma={}
    for t in range(len(trees)):
        for s in splits(trees,t):
            #create variables
            alpha[t,s]=0
            beta[t,s]=0
    for t in range(len(trees)):
        gamma[t]=0
        for l in leaves(trees,t):
            # The largest leaf of a tree is given to Gamma
            gamma[t] = max(gamma[t],prediction(trees,t,l,flag))
        
        
    def add_constraint(model, where):
        if where == GRB.Callback.MIPSOL:
        #sol_X = model.cbGetSolution([model._vars_X[i] for i in range(len(model._vars_X))])
            sol_theta=model.cbGetSolution([model._vars_theta[i] for i in range(len(model._vars_theta))])
            sol_X_one={}
            for i in total_split_variable(trees):
                for j in range(K(trees,i)):                
                    sol_X_one[i,j]=model.cbGetSolution(model._vars_X_one[i,j])
            alpha_new={}
            beta_new={}
            gamma_new={}
            # proposition 5
            for i in range(len(trees)):
                l_optimal=GETLEAF(trees,i,sol_X_one)
                for j in splits(trees,i):
                    temp1=0
                    if j in as_right_leaf(trees,i,l_optimal):
                        for l in left_leaf(trees,i,j):
                            temp1=max(temp1,(prediction(trees,i,l,flag)-prediction(trees,i,l_optimal,flag)))
                    alpha_new[i,j]=temp1
                    temp2=0
                    if j in as_left_leaf(trees,i,l_optimal):
                        for l in right_leaf(trees,i,j):
                            temp2=max(temp2,prediction(trees,i,l,flag)-prediction(trees,i,l_optimal,flag))
                    beta_new[i,j]=temp2
                gamma_new[i]=prediction(trees,i,l_optimal,flag)
            for i in reversed(range(len(trees))):
                expr=0
                for s in splits(trees, i):
                    expr=expr+alpha_new[i,s]*sol_X_one[V(trees,i,s),C(trees,i,s)]+beta_new[i,s]*(1-sol_X_one[V(trees,i,s),C(trees,i,s)])
                expr=expr+gamma_new[i]-sol_theta[i]            
                if expr < -1e-5:                
                    model.cbLazy(quicksum(alpha_new[i,s]*model._vars_X_one[V(trees,i,s),C(trees,i,s)] for s in splits(trees,i)) + quicksum(beta_new[i,s]*(1-model._vars_X_one[V(trees,i,s),C(trees,i,s)]) for s in splits(trees,i)) + gamma_new[i] - model._vars_theta[i] >= 0) 
                    print("Find a violated constraint!")
                    break 
                    #model.cbLazy(quicksum(alpha_new[i,s]*x(trees,model._vars_X,i,s) for s in splits(trees,i)) + quicksum(beta_new[i,s]*(1-x(trees,model._vars_X,i,s)) for s in splits(trees,i)) + gamma_new[i] - model._vars_theta[i] >= 0)


    #create a new model
    m = Model("tree_ensemble")

    #create variables
    X={}
    theta={}
    X_one={}

    for i in total_split_variable(trees):
        X[i]=m.addVar(lb[i], ub[i], name='X'+str(i))
        for j in range(K(trees,i)):
            X_one[i,j]=m.addVar(vtype=GRB.BINARY, name='X_one'+str(i)+'_'+str(j))
    for i in range(len(trees)):
        theta[i]=m.addVar(lb=-GRB.INFINITY, name='theta' + str(i))
    m.update()

    # Set objective
    m.setObjective(quicksum(lama[i]*theta[i] for i in range(len(trees))), GRB.MAXIMIZE)
    m.update()

    # Add constraint
    for i in range(len(trees)):
        m.addConstr(quicksum(alpha[i,s]*X_one[V(trees,i,s),C(trees,i,s)] for s in splits(trees,i)) + quicksum(beta[i,s]*(1-X_one[V(trees,i,s),C(trees,i,s)]) for s in splits(trees,i)) + gamma[i] - theta[i] >= 0) 

    for i in range(len(trees)):
        for j in splits(trees,i):
            m.addConstr((X_one[V(trees,i,j),C(trees,i,j)] == 1) >> (X[V(trees,i,j)] - split_values(trees,V(trees,i,j))[C(trees,i,j)] <= 0) )
            m.addConstr((X_one[V(trees,i,j),C(trees,i,j)] == 0) >> (X[V(trees,i,j)] - split_values(trees,V(trees,i,j))[C(trees,i,j)] >= 1e-5) )

    for i in total_split_variable(trees):
        for j in range(K(trees,i)-1):
            m.addConstr(X_one[i,j] - X_one[i,j+1] <= 0)

    m.update()

    m._vars_X_one=X_one
    m._vars_theta=theta
    m.params.LazyConstraints = 1
    m.optimize(add_constraint)
    
    #optimal_value=0
    #for i in range(len(trees)):
        #optimal_value=optimal_value+lama[i]*theta[i].x    

    #optimal_solution=np.zeros(len(X))
    #for i in range(len(X)):
        #optimal_solution[i]=X[i].x
    
    #return [optimal_value,optimal_solution]

def feasible_solution_solver(trees_given, flag_given, lb, ub):
    
    trees=trees_given
    flag=flag_given
    
    #create a new model
    m = Model("tree_ensemble_c_g")
    m.setParam('OutputFlag', 0)
    m.setParam('MIPFocus', 1)
    m.setParam("TimeLimit", 30)
    
    #create variables
    X={}
    Y_one={}
    Z={}
    T=total_split_variable(trees_given)
    L=len(trees)
    
    for i in T:
        X[i]=m.addVar(lb[i],ub[i], name='X'+str(i))
        for j in range(K(trees,i)):
            Y_one[i,j]=m.addVar(vtype=GRB.BINARY, name='Y_one'+str(i)+'_'+str(j))
    for t in range(L):
        for l in leaves(trees_given,t):
            Z[t,l]=m.addVar(lb=0, name='Z'+str(t)+'_'+str(l))
    m.update()
            
    # Set objective
    m.setObjective(quicksum(quicksum(prediction(trees,t,l,flag)*Z[t,l] for l in leaves(trees,t)) for t in range(L))+quicksum(X[i]*X[i] for i in T), GRB.MINIMIZE)
    m.update()

    # Add constraint 
    for t in range(L):
        #constraint 3b in GBT part
        m.addConstr(quicksum(Z[t,l] for l in leaves(trees,t)) ==1) 
                 
    # constraints 3c and 3d in GBT part
    for t in range(L):
        for s in splits(trees,t):
            m.addConstr(quicksum(Z[t,l] for l in left_leaf(trees,t,s)) <=Y_one[V(trees,t,s),C(trees,t,s)])
            m.addConstr(quicksum(Z[t,l] for l in right_leaf(trees,t,s)) <=1-Y_one[V(trees,t,s),C(trees,t,s)])
                   

    for i in T:
        #constraint 3e 
        for j in range(K(trees,i)-1):
            m.addConstr(Y_one[i,j] - Y_one[i,j+1] <= 0)
                   
    for i in range(L):
        for j in splits(trees,i):
            m.addConstr((Y_one[V(trees,i,j),C(trees,i,j)] == 1) >> (X[V(trees,i,j)] - split_values(trees,V(trees,i,j))[C(trees,i,j)] <= 0) )
            m.addConstr((Y_one[V(trees,i,j),C(trees,i,j)] == 0) >> (X[V(trees,i,j)] - split_values(trees,V(trees,i,j))[C(trees,i,j)] >= 1e-5) )


    m.update()

    m._vars_X=X
    m._vars_Y_one=Y_one
    m._vars_Z=Z
    m.optimize()
    
    optimal_value=m.objVal

    return optimal_value

def GBT_lowerbound_solver(trees_given, flag_given, lb, ub):
    
    trees=trees_given
    flag=flag_given 
         
    #create a new model
    m = Model("tree_ensemble_H")
    m.setParam('OutputFlag', 0)
    m.setParam("TimeLimit", 30)
 
    #create variables
    X={}
    Y_one={}
    Z={}
    
    T=total_split_variable(trees_given)
    L=len(trees)
    for i in T:
        X[i]=m.addVar(lb[i],ub[i], name='X'+str(i))
        for j in range(K(trees,i)):
            Y_one[i,j]=m.addVar(vtype=GRB.BINARY, name='Y_one'+str(i)+'_'+str(j))
    for t in range(L):
        for l in leaves(trees_given,t):
            Z[t,l]=m.addVar(lb=0, name='Z'+str(t)+'_'+str(l))
    m.update()
            
    # Set objective
    m.setObjective(quicksum(quicksum(prediction(trees,t,l,flag)*Z[t,l] for l in leaves(trees,t)) for t in range(L)), GRB.MINIMIZE)
    m.update()

    # Add constraint
    for t in range(L):
        #3b
        m.addConstr(quicksum(Z[t,l] for l in leaves(trees,t)) ==1) 
                 
    # 3c and 3d
    for t in range(L):
        for s in splits(trees,t):
            m.addConstr(quicksum(Z[t,l] for l in left_leaf(trees,t,s)) <=Y_one[V(trees,t,s),C(trees,t,s)])
            m.addConstr(quicksum(Z[t,l] for l in right_leaf(trees,t,s)) <=1-Y_one[V(trees,t,s),C(trees,t,s)])
        
                   
    for i in T:
        #3e
        for j in range(K(trees,i)-1):
            m.addConstr(Y_one[i,j] - Y_one[i,j+1] <= 0)
                   
    for i in range(L):
        for j in splits(trees,i):
            m.addConstr((Y_one[V(trees,i,j),C(trees,i,j)] == 1) >> (X[V(trees,i,j)] - split_values(trees,V(trees,i,j))[C(trees,i,j)] <= 0) )
            m.addConstr((Y_one[V(trees,i,j),C(trees,i,j)] == 0) >> (X[V(trees,i,j)] - split_values(trees,V(trees,i,j))[C(trees,i,j)] >= 1e-5) )


    m.update()

    m._vars_X=X
    m._vars_Y_one=Y_one
    m._vars_Z=Z
    m.optimize()
    
    optimal_value=m.objVal
 
    return optimal_value
def GBT_upperbound_solver(trees_given, flag_given, lb, ub):
    
    trees=trees_given
    flag=flag_given 
         
    #create a new model
    m = Model("tree_ensemble_GBT_upperbound")
    m.setParam('OutputFlag', 0)
 
    #create variables
    X={}
    Y_one={}
    Z={}
    
    T=total_split_variable(trees_given)
    L=len(trees)
    for i in T:
        X[i]=m.addVar(lb[i],ub[i], name='X'+str(i))
        for j in range(K(trees,i)):
            Y_one[i,j]=m.addVar(vtype=GRB.BINARY, name='Y_one'+str(i)+'_'+str(j))
    for t in range(L):
        for l in leaves(trees_given,t):
            Z[t,l]=m.addVar(lb=0, name='Z'+str(t)+'_'+str(l))
    m.update()
            
    # Set objective
    m.setObjective(quicksum(quicksum(prediction(trees,t,l,flag)*Z[t,l] for l in leaves(trees,t)) for t in range(L)), GRB.MAXIMIZE)
    m.update()

    # Add constraint
    for t in range(L):
        #3b
        m.addConstr(quicksum(Z[t,l] for l in leaves(trees,t)) ==1) 
                 
    # 3c and 3d
    for t in range(L):
        for s in splits(trees,t):
            m.addConstr(quicksum(Z[t,l] for l in left_leaf(trees,t,s)) <=Y_one[V(trees,t,s),C(trees,t,s)])
            m.addConstr(quicksum(Z[t,l] for l in right_leaf(trees,t,s)) <=1-Y_one[V(trees,t,s),C(trees,t,s)])
        
                   
    for i in T:
        #3e
        for j in range(K(trees,i)-1):
            m.addConstr(Y_one[i,j] - Y_one[i,j+1] <= 0)
                   
    for i in range(L):
        for j in splits(trees,i):
            m.addConstr((Y_one[V(trees,i,j),C(trees,i,j)] == 1) >> (X[V(trees,i,j)] - split_values(trees,V(trees,i,j))[C(trees,i,j)] <= 0) )
            m.addConstr((Y_one[V(trees,i,j),C(trees,i,j)] == 0) >> (X[V(trees,i,j)] - split_values(trees,V(trees,i,j))[C(trees,i,j)] >= 1e-5) )


    m.update()

    m._vars_X=X
    m._vars_Y_one=Y_one
    m._vars_Z=Z
    m.optimize()
    
    optimal_value=m.objVal
 
    return optimal_value


def get_GBT_lower_bound(tree_root_partition,flag,lb, ub):
    '''
    Get the GBT lower bound. The function is about lemma 2.
    '''
    # sum of all optimal values of trees within a partition subset
    lower_bound=0
    for i in range(len(tree_root_partition)):
        lower_bound=lower_bound+GBT_lowerbound_solver(tree_root_partition[i],flag,lb, ub)
    return lower_bound

def get_convex_bound(trees,lb,ub):
    
    '''
    Use gurobi to get the convex part objective value.    
    '''
      
    m = Model("convex_part")
    m.setParam('OutputFlag', 0)    
    X={}
    
    #add variables
    T=total_split_variable(trees)
    for i in T:
        X[i]=m.addVar(lb[i], ub[i], name='X'+str(i))
        
    #add objective function 
    m.setObjective(quicksum(X[i]*X[i] for i in T), GRB.MINIMIZE)
    m.optimize()
    
    #get the optimal value
    optimal_value=0
    for i in T:
        optimal_value=optimal_value+X[i].x*X[i].x
        
    return optimal_value



