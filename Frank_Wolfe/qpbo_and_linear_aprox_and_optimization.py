'''
Created on July 12, 2017

@author: UriS
How To build QPBO package for python:
Cython build :   cython --cplus pyqpbo.pyx
Package build (From MS C++ native prompt) :  python setup.py build_ext --inplace

Goal: Use QPBO optimizer and gradient decent method to assign marginal probabilities. Given:

1) All nodes can have state as [0, 1] only
2) Graph structure is given via J matrix described below

For optimization use QPBO Version 1.4 Copyright 2006-2008 Vladimir Kolmogorov (vnk@ist.ac.at)  

Structure:
    - Define Matrix J such that:
        - [i][i] element of matrix contains marginal probability for element i. Initialized proportionally to number of states (0.5 for two states model)
        - [i][j] element of matrix contains message from i to j (i->j). Initialized to [1,1...1]. Each message has same number of elements as states in node j
        - [i][j] is None in case node i is not connected to node j
    
Flow:

Example of usage. Define matrix for the tree below:
                0
              /  \
            1      2
          /  \    
        3     4

node_states=[[0,1],[0,1],[0,1],[0,1],[0,1]]

A=[[[0.5,0.5],[1,1],[1,1],None,None],
   [[1,1],[0.5,0.5],None,[1,1],[1,1]],
   [[1,1],None,[0.5,0.5],None,None],
   [None,[1,1],None,[0.5,0.5],None],
   [None,[1,1],None,None,[0.5,0.5]]]
'''
import numpy as np
import math
from copy import deepcopy
from pyqpbo import qpbo_solver_general_matrix_w_struct

nodes_states=[0,1]

DEBUG=False

def calc_x_log_y(x,y):
    if (x==0 or y==0):
        return 0
    else:
        return x*math.log(y)

def calc_x_log(x):
    if x<0.00001:
        #return float("inf")
        return -300000000000
    else:
        return math.log(x)

def phi_of_xi(w):
    #return np.exp(w)
    return 1

def message_function1(x,y):
    if (x+y) <= 1:
        return 1
    else:
        return 0

def message_function2(x,y):
    if x!=y:
        return 1
    else:
        return 0

class prettyfloat(float):
    def __repr__(self):
        return "%0.4f" % self
    
def print_m(J_matrix):
    print("Matrix is:")
    for i in range(len(J_matrix)):
        l=[]
        for mem in J_matrix[i]: 
            #print("mem type is..",type(mem))   
            if mem is None:
                l=l+["None"]
            elif type(mem)==float:
                l=l+[[prettyfloat(n) for n in mem]]
            elif type(mem)==int:
                l=l+[mem]
        print(l)
    #print(np.array(J_matrix))
    

def calculate_gradient_vector(J_matrix,node_states,message_function,neighbors_count):
    '''
    Calculate vector of all possible beliefs (diagonal items or nodes) including joint beliefs (non diagonal items or edges) 
    The vector holds beliefs for all possible node states and joint states
    Gradient vector is returned in the following form:
        [1,2,0,0] - node entry has zeros at the end as number of states
        [00,10,01,11] - edge entry in case nodes are connected
    [[1,2,0,0],[00,10,01,11],.....] 
    '''
    x=len(J_matrix[0])
    gradient_vector=[]
    J_flatten=[]
    array_index=0
    for i in range(x):
        for j in range(i,x):
            if i==j:
                gradient_vector.append([])
                for t,state in enumerate(node_states[i]):
                    partial_grad = calc_x_log(phi_of_xi(state))+(calc_x_log(J_matrix[i][i][t]))*(neighbors_count[i]-1) 
                    gradient_vector[array_index].append(partial_grad)
                    J_flatten.append(J_matrix[i][j][t])
                #TODO remove hardcoded
                gradient_vector[array_index].append(0)
                gradient_vector[array_index].append(0)
                array_index+=1
            elif (J_matrix[i][j] is not None):
                gradient_vector.append([])
                for t,state_i in enumerate(node_states[i]):
                    for s,state_j in enumerate(node_states[j]):
                        partial_grad = calc_x_log(message_function(state_i,state_j)) - (calc_x_log(J_matrix[i][j][t][s])+1)
                        gradient_vector[array_index].append(partial_grad)
                        J_flatten.append(J_matrix[i][j][t][s])
                array_index+=1
    if DEBUG:
        print('Grd vector after calculation ',gradient_vector)
        print('J_flatten after calc ', J_flatten)
    return gradient_vector,J_flatten

def calculate_node_state_prob(state, node_flag):
    '''
    Translate QPBO output for node into probability.
    If QPBO output is 0 and state is 0, than probability of state 0 is 1.
    In case QPBO returns -1, probability of each state is 0.5
    Dictionary: first element of key is state and second is output of QPBO
    '''
    state_flag_dict={(0,0):1, (0,1):0,
                     (1,0):0, (1,1):1,
                     (0,0.5):0.5, (1,0.5):0.5}
    return state_flag_dict[(state,node_flag)]

def calculate_edge_prob(state_i, state_j, node_i_flag, node_j_flag,message_function):
    '''
    Translate QPBO output for node i and j into marginal probability.
    Dictionary tuple: (state_i, state_j, node_i_flag, node_j_flag)
    '''
    state_flag_dict_a_zero={(0,0,1,1): 0, (0,1,1,1):0, (1,0,1,1):0, (1,1,1,1):1, #case QPBO flag_i =1 flag_j = 1
                            (0,0,0,1): 0, (0,1,0,1):1, (1,0,0,1):0, (1,1,0,1):0, #case QPBO flag_i =0 flag_j=1 
                            (0,0,0,0): 1, (0,1,0,0):0, (1,0,0,0):0, (1,1,0,0):0, #case QPBO flag_i =0 flag_j = 0 
                            (0,0,1,0): 0, (0,1,1,0):0, (1,0,1,0):1, (1,1,1,0):0, #case QPBO flag_i =1 flag_j=0
                            (0,0,1,0.5): 0, (0,1,1,0.5):0, (1,0,1,0.5):0.5, (1,1,1,0.5):0.5, #case QPBO flag_i = 1 and flag_j=0.5
                            (0,0,1,0.5): 0.5, (0,1,1,0.5):0.5, (1,0,1,0.5):0, (1,1,1,0.5):0, #case QPBO flag_i = 0 and flag_j=0.5
                            (0,0,0.5,1): 0, (0,1,0.5,1):0.5, (1,0,0.5,1):0, (1,1,0.5,1):0.5, #case QPBO flag_i = 0.5 and flag_j=1         
                            (0,0,0.5,1): 0.5, (0,1,0.5,1):0, (1,0,0.5,1):0.5, (1,1,0.5,1):0, #case QPBO flag_i = 0.5 and flag_j=0
                            (0,0,0.5,0.5): 0.5, (0,1,0.5,0.5):0, (1,0,0.5,0.5):0, (1,1,0.5,0.5):0.5,  #case QPBO flag_i = 0.5 and flag_j=0.5
                            }
    
    state_flag_dict_a_half={(0,0,1,1): 0, (0,1,1,1):0, (1,0,1,1):0, (1,1,1,1):1, #case QPBO flag_i =1 flag_j = 1
                            (0,0,0,1): 0, (0,1,0,1):1, (1,0,0,1):0, (1,1,0,1):0, #case QPBO flag_i =0 flag_j=1 
                            (0,0,0,0): 1, (0,1,0,0):0, (1,0,0,0):0, (1,1,0,0):0, #case QPBO flag_i =0 flag_j = 0 
                            (0,0,1,0): 0, (0,1,1,0):0, (1,0,1,0):1, (1,1,1,0):0, #case QPBO flag_i =1 flag_j=0
                            (0,0,1,0.5): 0, (0,1,1,0.5):0, (1,0,1,0.5):0.5, (1,1,1,0.5):0.5, #case QPBO flag_i = 1 and flag_j=0.5
                            (0,0,1,0.5): 0.5, (0,1,1,0.5):0.5, (1,0,1,0.5):0, (1,1,1,0.5):0, #case QPBO flag_i = 0 and flag_j=0.5
                            (0,0,0.5,1): 0, (0,1,0.5,1):0.5, (1,0,0.5,1):0, (1,1,0.5,1):0.5, #case QPBO flag_i = 0.5 and flag_j=1         
                            (0,0,0.5,1): 0.5, (0,1,0.5,1):0, (1,0,0.5,1):0.5, (1,1,0.5,1):0, #case QPBO flag_i = 0.5 and flag_j=0
                            (0,0,0.5,0.5): 0, (0,1,0.5,0.5):0.5, (1,0,0.5,0.5):0.5, (1,1,0.5,0.5):0,  #case QPBO flag_i = 0.5 and flag_j=0.5
                            }
    
    if (message_function(0,0)*message_function(1,1)) >= (message_function(0,1)*message_function(1,0)):
        prob = state_flag_dict_a_zero[(state_i, state_j, node_i_flag, node_j_flag)]
    else:
        prob = state_flag_dict_a_half[(state_i, state_j, node_i_flag, node_j_flag)]
    
    return prob
    
    
def calculate_marginal_prob_from_qpbo(nodes_sts,J_matrix, message_function):
    x=len(J_matrix[0])
    tao_vector=[]
    array_index=0
    for i in range(x):
        for j in range(i,x):
            if i==j:
                for t,state in enumerate(nodes_states):
                    state_prob = calculate_node_state_prob(state, nodes_sts[i]) 
                    tao_vector.append(state_prob)
                array_index+=1
            elif (J_matrix[i][j] is not None):
                for t,state_i in enumerate(nodes_states):
                    for s,state_j in enumerate(node_states[j]):
                        state_prob = calculate_edge_prob(state_i,state_j, nodes_sts[i],nodes_sts[j],message_function)
                        tao_vector.append(state_prob)
                        
                array_index+=1
    return tao_vector

def call_qpbo_optimizer(gradient_vector,J_matrix,n_nodes, n_edges, message_function):
    #gradient_vector=np.array(gradient_vector)
    sts = qpbo_solver_general_matrix_w_struct(gradient_vector,J_matrix, n_nodes,n_edges)
    #print('Result: ',sts)
    return calculate_marginal_prob_from_qpbo(sts,J_matrix, message_function)

def rebuild_probabilities_matrix(J_matrix, probabilities_arr,node_states):
    #print(J_matrix,"\n",probabilities_arr,"\n",node_states)
    x=len(J_matrix[0])
    array_index=0
    for i in range(x):
        for j in range(i,x):
            if i==j:
                for t,state in enumerate(node_states[i]):
                    J_matrix[i][j][t] = probabilities_arr[array_index]
                    array_index+=1
            elif (J_matrix[i][j] is not None):
                for t,state_i in enumerate(node_states[i]):
                    for s,state_j in enumerate(node_states[j]):
                        J_matrix[i][j][t][s] = probabilities_arr[array_index]
                        array_index+=1
    return J_matrix

def calculate_neighbors(J_matrix):
    '''
    Calculate number of edges for each node
    '''
    neighbors_count=np.zeros(len(J_matrix[0]))
    x=len(J_matrix[0])
    array_index=0
    for i in range(x):
        for j in range(i,x):
            if (i != j) and (J_matrix[i][j] is not None):
                neighbors_count[i]+=1
                neighbors_count[j]+=1
                
    return neighbors_count


def qpbo_gradient_optimizer(J_matrix,node_states,message_function):
    '''
    gradient_vector - has representation of [[node1-state0, node1-state1, 0, 0], edge-n0-n1[00,01, 10, 11], ....]
                    Only actual edges are present
    tao_t_vector - same as gradient vector, the only difference nodes don't have (0,0) padding 
    J_flatten - vector, contains probabilities for upper part of the matrix. Only not None edges present
    J_matrix - full matrix, with 'None' in case there is no edge between nodes
    
    '''
    n_nodes = len(J_matrix[0])
    n_edges = (sum(x is not None for x in J_matrix)-n_nodes)/2
    neighbors_count=calculate_neighbors(J_matrix)
    gradient_vector,J_flatten=calculate_gradient_vector(J_matrix,node_states,message_function1,neighbors_count)
    if DEBUG:
        print('Initial gradient vector', gradient_vector)
    
    print('J_flatten ',J_flatten)
    for t in range(5000):
        tao_t_vector = call_qpbo_optimizer(np.array(gradient_vector)*(-1),J_matrix,n_nodes,n_edges, message_function)
        if DEBUG:
            print("Probabilities after Linear optimizer..",tao_t_vector)
        alpha_t=2/(2+t)
        tao_t_plus_one=(1-alpha_t)*np.array(J_flatten) + alpha_t*np.array(tao_t_vector)
        if DEBUG:
            print(type(tao_t_plus_one),tao_t_plus_one[0],tao_t_plus_one)
        J_matrix=rebuild_probabilities_matrix(J_matrix,tao_t_plus_one,node_states)
        #print(J_matrix)
        gradient_vector,J_flatten=calculate_gradient_vector(J_matrix,node_states,message_function,neighbors_count)
    
    print(tao_t_plus_one)

'''******************************************************************************************************
   Start execution 
   ******************************************************************************************************
'''
print("")
print("***********************************************")
print("# Three node loop graph, weights=[1,1,1]")
J=[[[0.5,0.5],[[0.25,0.25],[0.25,0.25]],[[0.25,0.25],[0.25,0.25]]],
   [[[0.25,0.25],[0.25,0.25]],[0.5,0.5],[[0.25,0.25],[0.25,0.25]]],
   [[[0.25,0.25],[0.25,0.25]],[[0.25,0.25],[0.25,0.25]],[0.5,0.5]]]
weights=[1,1]
node_states=[[0,1],[0,1],[0,1]]
print("")
qpbo_gradient_optimizer(J,node_states,message_function1)



print("")
print("***********************************************")
print("# 5 Node tree (below), weights=[1,1,1]")
J=[
    [[0.5,0.5],[[0.25,0.25],[0.25,0.25]],[[0.25,0.25],[0.25,0.25]],None,None],
   [[[0.25,0.25],[0.25,0.25]],[0.5,0.5],None,[[0.25,0.25],[0.25,0.25]],[[0.25,0.25],[0.25,0.25]]],
   [[[0.25,0.25],[0.25,0.25]],None,[0.5,0.5],None,None],
   [None,[[0.25,0.25],[0.25,0.25]],None,[0.5,0.5],None],
   [None,[[0.25,0.25],[0.25,0.25]],None,None,[0.5,0.5]]
   ]
   
node_states=[[0,1],[0,1],[0,1],[0,1],[0,1]]
weights=[1,1]


print("")
qpbo_gradient_optimizer(J,node_states,message_function1)


'''
                0
              /  \
            1      2
          /  \    
        3     4

5000 IT
[ 0.71429102  0.28570898  0.57143611  0.14285491  0.28570898  0.
  0.35724559  0.35704543  0.28570898  0.          0.85714509  0.14285491
  0.42846895  0.42867614  0.14285491  0.          0.42846895  0.42867614
  0.14285491  0.          0.64295457  0.35704543  0.57132386  0.42867614
  0.57132386  0.42867614]
  
  [ 0.71436985  0.28563015  0.57126767  0.14310218  0.28563015  0.
  0.35708498  0.35728486  0.28563015  0.          0.85689782  0.14310218
  0.42835433  0.42854349  0.14310218  0.          0.42835433  0.42854349
  0.14310218  0.          0.64271514  0.35728486  0.57145651  0.42854349
  0.57145651  0.42854349]
'''