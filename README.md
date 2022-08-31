# TREE ENSEMBLE OPTIMIZATION PROBLEM
Optimization on Tree Ensembles.

This project is written in Python, and uses Gurobi as an optimization solver.

This project takes tree ensemble as input and uses the Benders Decomposition method to solve the large-scale optimization problem. Please check the paper "Optimization of Tree Ensemble", i.e., Optimization of Tree Ensembles.pdf file, for more details about the method. And the project also investigates the GBT lower bounding method and Branch and bound algorithm in the paper " Mixed-Integer Convex Nonlinear Optimization with Gradient-Boosted Trees Embedded" (the pdf file). Please check the pdf file for more information.

First, we use random forests regression and classifiers in the project. Sometimes we also use the Gradient Boosting regression or classifier trees for some tests in the jupyter notebooks. And we use the concrete.csv for regression problems and generate some data and using some real datasets for classification.

Second, the structure of the project is as follows:

`prepareziyue.py` is written to process the tree ensemble as input, including how to find the leaf and its prediction, the left-child tree, and so on. And some other functions such as the nonroot partition refinement method function and root partition method function are added. Please check the " Mixed-Integer Convex Nonlinear Optimization with Gradient-Boosted Trees Embedded"(the pdf file) for more information about the two methods.

`ziyuesolver_2.py`  is written to solve the optimization problem calling Gurobi in Python. And function MIP_solver comes from the project `TREE-master` written for the paper "Optimization of Tree Ensemble". And functions `feasible_solution_solver`, `GBT_lowerbound_solver` and `GBT_upperbound_solver` are added to solve the tree optimization problem and GBT part of the problem. Note: We use random forests regression or classifier in the project, but we also call the functions `GBT...`  to be consistent with the method names in the paper " Mixed-Integer Convex Nonlinear Optimization with Gradient-Boosted Trees Embedded".

`Node.py`  is the preparation for Branch and bound algorithm. We define a class "Node" that has some attributes and methods such as lower bound vector and upper bound vector for the domain of the node, `update_tree_partition` function, `get_GBT_obj_val` function, and so on. There is also a function `get_children` to get the children of the node. Please check the `Node.py`  for more details.

`BB.py`  is the Branch and bound algorithm. We define some functions such as `weight_calculation`, `Strong_branching`, and `B_B` for the Branch and bound algorithm. Please check the paper " Mixed-Integer Convex Nonlinear Optimization with Gradient-Boosted Trees Embedded" (the pdf file) for more information about the weight calculation of split nodes and strong branching.

`MIO_Four_Example.ipynb`,`GBT_Lower_bounding.ipynb` and `B&B.ipynb` are reports about numerical experiment tests of the project. 


