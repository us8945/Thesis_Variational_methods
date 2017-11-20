'''
Created on July 12, 2017

@author: UriS
How To build QPBO package for python:
Cython build :   cython --cplus pyqpbo.pyx
Package build (From MS C++ native prompt) :  python setup.py build_ext --inplace

Need numpy version 1.13 in order to use np.unique with axis

Change log:
version v1: add support for regularization and edge weigts constants

Goal: 
Learn graph parameters given:
- Graph structure 
- Sample from population of size M 

Output: node weights and edges potential functions


Use QPBO optimizer and gradient decent method to assign marginal probabilities. Given:

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
import cvxopt
import math
from copy import deepcopy
from scipy.constants.constants import alpha
from scipy.optimize import linprog
from pyqpbo import qpbo_solver_general_matrix_w_struct
import cProfile

from optimization import gibs_sampling

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

def calculate_b_i_j(x_i,x_j,i,j,J_matrix,col_weigths,states,message_function):
    '''
    Calculate join beliefs for node-i and node-j. Example:
    if both have states [0,1], there will be 4 combinations of state-i and state-j that will have belief calculated:
    0-0
    0-1
    1-0
    1-1 
    '''
    messages_to_i = incoming_messages(i,len(states[i]),j,J_matrix)[x_i] #all incoming messages to i exclude message from j
    messages_to_j = incoming_messages(j,len(states[i]),i,J_matrix)[x_j] #all incoming messages to j exclude message from i
    bij=phi_of_xi(col_weigths[x_i])*phi_of_xi(col_weigths[x_j])*message_function(x_i,x_j)*messages_to_i*messages_to_j
    return bij
    
def calculate_b_i_log(J_matrix,states):
    ''' Used to calculate one element for Beth free energy 
    '''
    nodes_count=len(J_matrix[0])
    Sum_bi_log_bi=0
    for i in range(nodes_count):
        for s in range(len(states[i])):
            #print(J_matrix[i][i][s],Z_vector[i])
            b_i=J_matrix[i][i][s]
            Sum_bi_log_bi=Sum_bi_log_bi+calc_x_log_y(b_i,b_i)
            #print("Sum_bi_log_bi: ",Sum_bi_log_bi,b_i,calc_x_log_y(b_i,b_i))
    
    #print("Sum_bi_log_bi: ", Sum_bi_log_bi)
    return (-1)*Sum_bi_log_bi

def calc_bi_log_phi(J_matrix,states,col_weigths):
    nodes_count=len(J_matrix[0])
    Sum_bi_log_phi=0
    for i in range(nodes_count):
        for s in range(len(states[i])):
            #print(J_matrix[i][i][s],Z_vector[i])
            b_i=J_matrix[i][i][s]
            Sum_bi_log_phi=Sum_bi_log_phi+b_i*math.log(phi_of_xi(col_weigths[s]))
            #print("Sum_bi_log_phi: ",Sum_bi_log_phi,b_i,math.log(phi_of_xi(col_weigths[s])))
    
    #print("calc_bi_log_phi: ", Sum_bi_log_phi)
    return Sum_bi_log_phi

def calc_b_ij_psi(J_matrix,col_weigths,states,message_function):
    nodes_count=len(J_matrix[0])
    Sum_bij_log_bij=0
    for i in range(nodes_count):
        for j in range(i+1,nodes_count): #Go above diagonal of the matrix, enumerate over edges 
            if(J_matrix[i][j] is not None): #Skip in case nodes are not connected
                norm=0
                for s_i in range(len(states[i])):
                    for s_j in range(len(states[j])):
                        b_i_j=calculate_b_i_j(s_i,s_j,i,j,J_matrix,col_weigths,states,message_function)
                        norm+=b_i_j
                for s_i in range(len(states[i])):
                    for s_j in range(len(states[j])):
                        #print(J_matrix[i][i][s],Z_vector[i])
                        b_i=J_matrix[i][i][s_i]
                        b_j=J_matrix[j][j][s_j]
                        b_i_j=calculate_b_i_j(s_i,s_j,i,j,J_matrix,col_weigths,states,message_function)
                        b_i_j=b_i_j/norm
                        #print("Sum_bij_log_bij: ",i,j,s_i,s_j,b_i,b_j,b_i_j)
                        #print(b_i_j,s_i,s_j,message_function(s_i,s_j))
                        Sum_bij_log_bij=Sum_bij_log_bij+calc_x_log_y(b_i_j,message_function(s_i,s_j))
    
    return Sum_bij_log_bij

def calculate_b_ij_log(J_matrix,col_weigths,states,message_function):
    nodes_count=len(J_matrix[0])
    Sum_bij_log_bij=0
    for i in range(nodes_count):
        for j in range(i+1,nodes_count): #Go above diagonal of the matrix, enumerate over edges 
            if(J_matrix[i][j] is not None): #Skip in case nodes are not connected
                norm=0
                for s_i in range(len(states[i])):
                    for s_j in range(len(states[j])):
                        b_i_j=calculate_b_i_j(s_i,s_j,i,j,J_matrix,col_weigths,states,message_function)
                        norm+=b_i_j
                for s_i in range(len(states[i])):
                    for s_j in range(len(states[j])):
                        #print(J_matrix[i][i][s],Z_vector[i])
                        b_i=J_matrix[i][i][s_i]
                        b_j=J_matrix[j][j][s_j]
                        b_i_j=calculate_b_i_j(s_i,s_j,i,j,J_matrix,col_weigths,states,message_function)
                        b_i_j=b_i_j/norm
                        #print("Sum_bij_log_bij: ",i,j,s_i,s_j,b_i,b_j,b_i_j)
                        #print(b_i_j/(b_i*b_j))
                        Sum_bij_log_bij=Sum_bij_log_bij+calc_x_log_y(b_i_j,(b_i_j/(b_i*b_j)))
    #print("Sum_bij_log_bij: ",Sum_bij_log_bij)
    return (-1)*Sum_bij_log_bij
    
def calculate_Z(J_matrix,states,message_function,col_weigths):
    ''' To do: add second element for general case
    '''
    log_Z = calc_bi_log_phi(J_matrix,states,col_weigths) + \
            calc_b_ij_psi(J_matrix,col_weigths,states,message_function) + \
            calculate_b_i_log(J_matrix,states)+ \
            calculate_b_ij_log(J_matrix,col_weigths,states,message_function)
    return math.exp(log_Z)

    
def calculate_message(x_states,y_states,message_to_x,message_function,normalize_ind=True):
    message=[]
    message_out=[]
    for y in y_states:
        for x in x_states:
            message.append(message_function(x,y))
        #print('Before',message_to_x,message)
        message_out.append(np.dot(message_to_x,message))
        message=[]
    if (normalize_ind==True):
        message_out=normalize_message(message_out)
        
    return message_out

def normalize_message(message):
    #return message
    z_i = np.sum(message)
    return np.multiply(message,1/z_i)

def incoming_messages(i_node,message_dim,exclude,J_matrix):
    message=np.ones(message_dim)
    for j in range(len(J_matrix)):
        if j!=i_node and j!=exclude and J_matrix[j][i_node] is not None:
            #print(message, J_matrix[j][i_node])
            message = np.multiply(message,J_matrix[j][i_node])
    
    #print("Message into node :",i_node,message)
    return message


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

#def 


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
    
        
def pass_over_matrix(col_weigths,J_matrix,node_states,message_function,iterations=50):
    x=len(J_matrix[0])
    for k in range(iterations):
        for i in range(x):
            for j in range(x):
                message_to_i = incoming_messages(i,len(node_states[i]),j,J_matrix) #all incoming message exclude message from j
                if i!=j and J_matrix[i][j] is not None: #send message from i to j
                    m = calculate_message(node_states[i],node_states[j],message_to_i,message_function,True)
                    J_matrix[i][j]= list(np.multiply(m,phi_of_xi(col_weigths))) #Multiply message by state weight and place into matrix np.multiply([2,3],[3,4])=[6,12]
    for i in range(x):         
        message_to_i = incoming_messages(i,len(node_states[i]),None,J_matrix)
        b_i=normalize_message(list(np.multiply(message_to_i.tolist(),phi_of_xi(col_weigths))))
        J_matrix[i][i]= b_i

def sumprod(A,col_weigths,node_states,message_function,its):
    '''
    A - graph matrix 
    w - vector of node weights
    its - number of iterations
    Returns Z - aproximate partition function 
    '''
    print("Node states: ",node_states)
    print("Weights: ", col_weigths)
    pass_over_matrix(col_weigths,A,node_states,message_function,50)
    print_m(A)
    return calculate_Z(A,node_states,message_function,col_weigths)


def calculate_gradient_vector(J_matrix,node_states,graph_parameters,edges_weights_sum):
    '''
    J_matrix - contains graph structure as well as probabilities from previous round
    Calculate vector of all possible beliefs (diagonal items or nodes) including joint beliefs (non diagonal items or edges) 
    The vector holds beliefs for all possible node states and joint states
    Gradient vector is returned in the following form:
        [1,2,0,0] - node entry has zeros at the end as number of states
        [00,01,10,11] - edge entry in case nodes are connected
    [[1,2,0,0],[00,10,01,11],.....] 
    '''
    x=len(J_matrix[0])
    gradient_vector=[]
    J_flatten=[]
    array_index=0
    grph_param_ind=0
    for i in range(x):
        for j in range(i,x):
            if i==j:
                gradient_vector.append([])
                for t,state in enumerate(node_states):
                    #print('i:',i,'j:',j,'state:',state)
                    #TODO validate plus 1 in second term
                    partial_grad = calc_x_log(graph_parameters[grph_param_ind])+(calc_x_log(J_matrix[i][i][t])+1)*(edges_weights_sum[i]-1) 
                    #partial_grad = calc_x_log(graph_parameters[grph_param_ind])+(calc_x_log(J_matrix[i][i][t]))*(neighbors_count[i]-1)
                    gradient_vector[array_index].append(partial_grad)
                    grph_param_ind+=1
                    J_flatten.append(J_matrix[i][j][t])
                #TODO remove hardcoded
                gradient_vector[array_index].append(0)
                gradient_vector[array_index].append(0)
                array_index+=1
            elif (J_matrix[i][j] is not None):
                gradient_vector.append([])
                for t,state_i in enumerate(node_states):
                    for s,state_j in enumerate(node_states):
                        partial_grad = calc_x_log(graph_parameters[grph_param_ind]) - (calc_x_log(J_matrix[i][j][t][s])+1)
                        gradient_vector[array_index].append(partial_grad)
                        J_flatten.append(J_matrix[i][j][t][s])
                        #print('i:',i,'j:',j,'state_i:',state_i,'state_j:',state_j)
                        grph_param_ind+=1
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

def calculate_edge_prob(state_i, state_j, node_i_flag, node_j_flag,edge_parameters):
    '''
    Translate QPBO output for node i and j into marginal probability.
    Dictionary tuple: (state_i, state_j, node_i_flag, node_j_flag)
    '''
    state_flag_dict_a_zero={(0,0,1,1): 0, (0,1,1,1):0, (1,0,1,1):0, (1,1,1,1):1, #case QPBO flag_i =1 flag_j = 1
                            (0,0,0,1): 0, (0,1,0,1):1, (1,0,0,1):0, (1,1,0,1):0, #case QPBO flag_i =0 flag_j=1 
                            (0,0,0,0): 1, (0,1,0,0):0, (1,0,0,0):0, (1,1,0,0):0, #case QPBO flag_i =0 flag_j = 0 
                            (0,0,1,0): 0, (0,1,1,0):0, (1,0,1,0):1, (1,1,1,0):0, #case QPBO flag_i =1 flag_j=0
                            (0,0,1,0.5): 0, (0,1,1,0.5):0, (1,0,1,0.5):0.5, (1,1,1,0.5):0.5, #case QPBO flag_i = 1 and flag_j=0.5
                            (0,0,0,0.5): 0.5, (0,1,0,0.5):0.5, (1,0,0,0.5):0, (1,1,0,0.5):0, #case QPBO flag_i = 0 and flag_j=0.5
                            (0,0,0.5,1): 0, (0,1,0.5,1):0.5, (1,0,0.5,1):0, (1,1,0.5,1):0.5, #case QPBO flag_i = 0.5 and flag_j=1         
                            (0,0,0.5,0): 0.5, (0,1,0.5,0):0, (1,0,0.5,0):0.5, (1,1,0.5,0):0, #case QPBO flag_i = 0.5 and flag_j=0
                            (0,0,0.5,0.5): 0.5, (0,1,0.5,0.5):0, (1,0,0.5,0.5):0, (1,1,0.5,0.5):0.5,  #case QPBO flag_i = 0.5 and flag_j=0.5
                            }
    
    state_flag_dict_a_half={(0,0,1,1): 0, (0,1,1,1):0, (1,0,1,1):0, (1,1,1,1):1, #case QPBO flag_i =1 flag_j = 1
                            (0,0,0,1): 0, (0,1,0,1):1, (1,0,0,1):0, (1,1,0,1):0, #case QPBO flag_i =0 flag_j=1 
                            (0,0,0,0): 1, (0,1,0,0):0, (1,0,0,0):0, (1,1,0,0):0, #case QPBO flag_i =0 flag_j = 0 
                            (0,0,1,0): 0, (0,1,1,0):0, (1,0,1,0):1, (1,1,1,0):0, #case QPBO flag_i =1 flag_j=0
                            (0,0,1,0.5): 0, (0,1,1,0.5):0, (1,0,1,0.5):0.5, (1,1,1,0.5):0.5, #case QPBO flag_i = 1 and flag_j=0.5
                            (0,0,0,0.5): 0.5, (0,1,0,0.5):0.5, (1,0,0,0.5):0, (1,1,0,0.5):0, #case QPBO flag_i = 0 and flag_j=0.5
                            (0,0,0.5,1): 0, (0,1,0.5,1):0.5, (1,0,0.5,1):0, (1,1,0.5,1):0.5, #case QPBO flag_i = 0.5 and flag_j=1         
                            (0,0,0.5,0): 0.5, (0,1,0.5,0):0, (1,0,0.5,0):0.5, (1,1,0.5,0):0, #case QPBO flag_i = 0.5 and flag_j=0
                            (0,0,0.5,0.5): 0, (0,1,0.5,0.5):0.5, (1,0,0.5,0.5):0.5, (1,1,0.5,0.5):0,  #case QPBO flag_i = 0.5 and flag_j=0.5
                            }
    
    #print('Edge params:', edge_parameters)
    #prob = state_flag_dict_a_zero[(state_i, state_j, node_i_flag, node_j_flag)]
    
    if (edge_parameters[0]*edge_parameters[3]) >= (edge_parameters[1]*edge_parameters[2]): #if 00*11 >=01*10
        prob = state_flag_dict_a_zero[(state_i, state_j, node_i_flag, node_j_flag)]
        #print('Edge0:',edge_parameters,(state_i, state_j, node_i_flag, node_j_flag),'Prob:',prob)
    else:
        prob = state_flag_dict_a_half[(state_i, state_j, node_i_flag, node_j_flag)]
        #print('Edge1:',edge_parameters,(state_i, state_j, node_i_flag, node_j_flag),'Prob:',prob)
    
    return prob
    
    
def calculate_marginal_prob_from_qpbo(nodes_sts,J_matrix, graph_parameters,node_states):
    '''
    nodes_sts - output from QPBO
    graph_parameters - model parameters
    node_states - valid node states, in our case [0,1]
    '''
    x=len(J_matrix[0])
    tao_vector=[]
    array_index=0
    grph_param_ind=0
    for i in range(x):
        for j in range(i,x):
            if i==j:
                for t,state in enumerate(node_states):
                    state_prob = calculate_node_state_prob(state, nodes_sts[i]) 
                    tao_vector.append(state_prob)
                    grph_param_ind+=1
                    #print('Advance grph_param_ind',grph_param_ind)
                array_index+=1
            elif (J_matrix[i][j] is not None):
                #print('Edge index start',grph_param_ind,'i=',i,'j=',j)
                edge_parameters=graph_parameters[grph_param_ind:grph_param_ind+4] #In case of {0,1} states we have 00,01,10,11 combinations
                #print('Edge parameters',type(edge_parameters))
                for t,state_i in enumerate(node_states):
                    for s,state_j in enumerate(node_states):
                        state_prob = calculate_edge_prob(state_i,state_j, nodes_sts[i],nodes_sts[j],edge_parameters)
                        tao_vector.append(state_prob)
                        grph_param_ind+=1
                        
                array_index+=1
    return tao_vector

def call_qpbo_optimizer(gradient_vector,J_matrix,n_nodes, n_edges, graph_parameters,node_states):
    #gradient_vector=np.array(gradient_vector)
    sts = qpbo_solver_general_matrix_w_struct(gradient_vector,J_matrix, n_nodes,n_edges)
    if DEBUG:
        print('Result: ',sts)
    return calculate_marginal_prob_from_qpbo(sts,J_matrix, graph_parameters,node_states)

def rebuild_probabilities_matrix(J_matrix, probabilities_arr,node_states):
    #print(J_matrix,"\n",probabilities_arr,"\n",node_states)
    x=len(J_matrix[0])
    array_index=0
    for i in range(x):
        for j in range(i,x):
            if i==j:
                for t,state in enumerate(node_states):
                    J_matrix[i][j][t] = probabilities_arr[array_index]
                    array_index+=1
            elif (J_matrix[i][j] is not None):
                for t,state_i in enumerate(node_states):
                    for s,state_j in enumerate(node_states):
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


def qpbo_gradient_optimizer(J_matrix,node_states,graph_parameters,edges_weights_sum):
    '''
    gradient_vector - has representation of [[node1-state0, node1-state1, 0, 0], edge-n0-n1[00,01, 10, 11], ....]
                    Only actual edges are present
    tao_t_vector - same as gradient vector, the only difference nodes don't have (0,0) padding 
    J_flatten - vector, contains probabilities for upper part of the matrix. Only not None edges present
    J_matrix - full matrix, with 'None' in case there is no edge between nodes
    graph_parameters - vector of node weights and edge functions. The order corresponds to J_matrix upper diagonal part order
    
    '''
    iterations=10
    n_nodes = len(J_matrix[0])
    #n_edges = (sum(x is not None for x in J_matrix)-n_nodes)/2
    #TODO Fix number of edges
    #TODO Move neighbors calculation to be parameter of the QPBO optimizer
    n_edges=1
    neighbors_count=calculate_neighbors(J_matrix)
    if DEBUG:
        print('QPBO inputs:********************************')
        print('Matrix ',J_matrix)
        print('neighbors_count', neighbors_count)
        print('Number of edges:',n_edges)
        print('Number of nodes:',n_nodes)
    gradient_vector,J_flatten=calculate_gradient_vector(J_matrix,node_states,graph_parameters,edges_weights_sum)
    if DEBUG:
        print('Initial gradient vector', gradient_vector)
    #J_flatten = flatten_matrix(J_matrix)
    
    if DEBUG:
        print('J_flatten ',J_flatten)
    for t in range(iterations):
        tao_t_vector = call_qpbo_optimizer(np.array(gradient_vector)*(-1),J_matrix,n_nodes,n_edges, graph_parameters,node_states)
        if DEBUG:
            print("Probabilities after QPBO optimizer..",tao_t_vector)
        alpha_t=2/(2+t)
        tao_t_plus_one=(1-alpha_t)*np.array(J_flatten) + alpha_t*np.array(tao_t_vector)
        if DEBUG:
            print('tao_t_plus_one:',tao_t_plus_one)
        J_matrix=rebuild_probabilities_matrix(J_matrix,tao_t_plus_one,node_states)
        #print(J_matrix)
        gradient_vector,J_flatten=calculate_gradient_vector(J_matrix,node_states,graph_parameters,edges_weights_sum)
    
    if DEBUG:
        print('Graph parameters at the end;',graph_parameters)
        print('Tao t-plus one at the end:',tao_t_plus_one)
    return tao_t_plus_one

def calc_i_j_combinations(d_matrix):
    '''
    Returns array of number of combinations found for [0,0],[0,1], [1,0],[1,1]    
    '''
    return_aray=[0,0,0,0]
    comb,counts=np.unique(d_matrix,axis=0,return_counts=True)
    #print(comb,len(comb))
    for i in range(len(comb)):
        if np.array_equal(comb[i],[0,0]):
            return_aray[0]=counts[i]
        elif np.array_equal(comb[i],[0,1]):
            return_aray[1]=counts[i]
        elif np.array_equal(comb[i],[1,0]):
            return_aray[2]=counts[i]
        elif np.array_equal(comb[i],[1,1]):
            return_aray[3]=counts[i]
    
    return return_aray



def calc_emp_probabilities(samples, J_matrix):
    '''
    Input: Samples in the format of np.matrix. Each column is one sample. Transform samples to the following:
               - samples in the format of array of arrays. Each array represent one record from sample.
               Each member of the record is value of the node
               Example: [[1,1,1],[1,0,1],[0,0,0]] . 
           Represent node values of graph with three nodes. First record, all nodes have value of 1. 
    
           J_matrix - graph representation, used to calculate marginals only for relevant edges
    
    Output
    '''
    #d[:,[0,2]] - slice first and 3rd element of every record
    #np.unique(d[:,[0,2]],axis=0,return_counts=True) - return combinations and counts
    np_samples = np.array(samples).transpose()
    number_of_samples=len(np_samples)
    emp_probabilities=[]
    x=len(J_matrix[0])
    nodes_states=[0,1]
    nodes_average_one_count = np.mean(np_samples,axis=0) #Average number of 1's in each column/node
    #tao_vector=[]
    for i in range(x):
        for j in range(i,x):
            if i==j: 
                emp_probabilities.append(1-nodes_average_one_count[i]) # Add probability for 0
                emp_probabilities.append(nodes_average_one_count[i])   # Add probability for 1
            elif (J_matrix[i][j] is not None):
                state_prob=calc_i_j_combinations(np_samples[:,[i,j]])  # Calculate number of combinations between node i and j
                if DEBUG:
                    #print('Marginals',i,j,state_prob)
                    print('Marginals','node_i:',i,'node_j:',j,'counts:',state_prob,'prob:',np.array(state_prob)/number_of_samples)
                state_prob=np.array(state_prob)/number_of_samples
                for s in range(len(state_prob)):
                    emp_probabilities.append(state_prob[s])
                        
    return emp_probabilities

def calc_edges_counts(J_matrix,edges_weight):
    '''
    Input: array of weight per edge
    Output: array of sum of weights for incoming edges per node
    '''
    sum_edges=np.zeros(len(J_matrix[0]))
    ind=0
    for i in range(len(J_matrix[0])):
        for j in range(i+1,len(J_matrix[0])):
            if J_matrix[i][j] is not None:
                sum_edges[i]+=edges_weight[ind]
                sum_edges[j]+=edges_weight[ind]
                ind+=1
    
    return sum_edges
    
def learn_parameters(J_matrix, samples, node_states,edges_weight,regularization=0,iterations=100):
    '''
    Input: 
          J_matrix - describes graph
          samples  - samples to estimate model parameters 
    
    Output:
         Model parameters
    
    Flow:
       a) calculate empirical probabilities from samples
       b) use empirical probabilities to run qpbo_gradient_optimizer and get update probabilities and marginal probabilities
       c) use probabilities from (b) to calculate gradient and update next round parameters
       d) return to (b)
        
    '''
    DEBUG_1=False
    if DEBUG_1:
        print('Learn params Input Matrix: ', J_matrix)
    empirical_prob = np.array(calc_emp_probabilities(samples,J_matrix))
    #print('Empiracal probabilities, model input:',empirical_prob)
    if DEBUG_1:
        print('Inside Learn params, input sample empirical probabilities: ', empirical_prob)
    prev_parameters = np.ones(len(empirical_prob))#*math.exp(1)
    
    #Calculate summary of incoming edges weights per node. Returns array of sum per node
    edges_weights_sum=calc_edges_counts(J_matrix,edges_weight)
    print('Summ of edges weights:',edges_weights_sum)
    for i in range(iterations):
        ''' Adding np.exp() '''
        #calc_probability = qpbo_gradient_optimizer(J_matrix,node_states,prev_parameters)
        calc_probability = qpbo_gradient_optimizer(J_matrix,node_states,np.exp(prev_parameters),edges_weights_sum)
        if DEBUG_1:
            print('Round:',i)
            print('Emp prob:                :',empirical_prob)
            print('Calculated by QBPO probab:',calc_probability)
        gradient = empirical_prob - calc_probability
        if DEBUG_1:
            print('Gradient:',gradient)
        new_parameters = prev_parameters + (2/(2+i))*(gradient - regularization*prev_parameters)
        if DEBUG_1:
            print('Prev parameters',prev_parameters)
            print('New parameters',new_parameters)
        prev_parameters = new_parameters
        if i%25==0:
            print('Iteration, params',i,prev_parameters)
        
    return prev_parameters



if __name__ == '__main__':

    '''******************************************************************************************************
       Start execution 
       ******************************************************************************************************
    '''
    
    '''
                0
              /  \
            1      2
    '''
    RUN_1=True
    J=[[[1,1],[[1,1],[1,1]],[[1,1],[1,1]]],
       [[[1,1],[1,1]],[1,1],None],
       [[[1,1],[1,1]],None,[1,1]]]
    
    graph_parameters = [ 1,1,   1,2,2,1, 1,2,2,1,   1,1,  1,1]
    graph_parameters=np.exp(graph_parameters)
    
    if RUN_1:
        qpbo_probs=qpbo_gradient_optimizer(J,[0,1],graph_parameters)
        print('Output of QPBO:', qpbo_probs)
    
    else:
        print("")
        print("***********************************************")
        print("# Three node loop graph")
        J=[[[0.5,0.5],[[0.25,0.25],[0.25,0.25]],[[0.25,0.25],[0.25,0.25]]],
           [[[0.25,0.25],[0.25,0.25]],[0.5,0.5],[[0.25,0.25],[0.25,0.25]]],
           [[[0.25,0.25],[0.25,0.25]],[[0.25,0.25],[0.25,0.25]],[0.5,0.5]]]
        weights=[1,1]
        #node_states=[[0,1],[0,1],[0,1]]
        print("")
        #qpbo_gradient_optimizer(J,node_states,message_function1)
        
        '''
        s=[[1,1,1],[1,0,1],[0,0,0],[0,0,0],[1,1,0]]
        emp_prob = calc_emp_probabilities(s,J)
        print('Emp probab ',len(emp_prob), emp_prob)
        node_states=[0,1]
        graph_params = learn_parameters(J,s)
        print('Learned params: ')
        print(graph_params)
        print("")
        
        
        '''
        print("")
        print("***********************************************")
        print("# 5 Node tree (below)")
        J=[
            [[0.5,0.5],[[0.25,0.25],[0.25,0.25]],[[0.25,0.25],[0.25,0.25]],None,None],
           [[[0.25,0.25],[0.25,0.25]],[0.5,0.5],None,[[0.25,0.25],[0.25,0.25]],[[0.25,0.25],[0.25,0.25]]],
           [[[0.25,0.25],[0.25,0.25]],None,[0.5,0.5],None,None],
           [None,[[0.25,0.25],[0.25,0.25]],None,[0.5,0.5],None],
           [None,[[0.25,0.25],[0.25,0.25]],None,None,[0.5,0.5]]
           ]
        node_states=[0,1]
        s=[[1,0,0,1,1],[1,0,1,0,0],[0,0,0,0,0],[0,0,0,1,1],[1,1,0,1,0]]
        ns=np.matrix(s).transpose()
        print('Old emp prob:',len(calc_emp_probabilities(s,J)),calc_emp_probabilities(s,J))
        emp_prob = calc_emp_probabilities(ns,J)
        print('Emp probab ',len(emp_prob), emp_prob)
        #parameters = qpbo_gradient_optimizer(J,node_states,emp_prob)
        #print('First round parameters', type(parameters))
        graph_params = learn_parameters(J,ns,node_states)
        print('Learned params: ')
        print(graph_params)
        print("")
        samples = gibs_sampling.gibbs(J,graph_params,2**10, 2**10)
        
        empirical_prob = np.array(calc_emp_probabilities(samples,J))
        
        print('New probabilities', empirical_prob)
    
        #qpbo_gradient_optimizer(J,node_states,message_function1)
        
        
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