'''
Created on March 11, 2017

@author: Uri Smashnov

Matrix definition:
    - Define Matrix J such that:
        - [i][i] element of matrix contains marginal probability for element i. Initialized proportionally to number of states (0.5 for two states model)
        - [i][j] element of matrix contains message from i to j (i->j). Initialized to [1,1...1]. Each message has same number of elements as states in node j
        - [i][j] is None in case node i is not connected to node j
    - Define vector H to hold weight for each node. Number of elements as number of nodes
    - Define array States to hold list of states for each node

Flow:
1) To determine initial assignment and matrix of max-marginals : Run max-prod BP algorithm on input matrix, color weights and number of iterations
2) Run sum-product (BP) to calculate correct beliefs, and store in A matrix
2) Run Gibbs sampling for number of burn-in iterations
3) Run Gibbs sampling for number of "its" iterations
3) Create vector of vectors of color assignments for each node
4) Output matrix of marginals

'''
import numpy as np
import math
import random
from itertools import groupby
from copy import deepcopy
from random import randint
import cProfile
import copy
from optimization import learn_parameters_qpbo_optimization_v1, verify_model_v1
import itertools

DEBUG=False

def print_m(J_matrix):
    '''Utility function to print matrix
    '''
    print("Matrix is:")
    #for i in range(len(J_matrix)):
    #    print(J_matrix[i])
    
    for i in range(len(J_matrix)):
        l=[]
        for mem in J_matrix[i]:    
            if mem is None:
                l=l+["None"]
            else:
                l=l+[[prettyfloat(n) for n in mem]]
        print(l)
    #print(np.array(J_matrix))    

class prettyfloat(float):
    def __repr__(self):
        return "%0.4f" % self


def normalize_message(message):
    #return message
    z_i = np.sum(message)    
    return np.multiply(message,1/z_i)


def calculate_state_given_states(x_node,A,node_states,current_states):
    '''
    Calculate x_node states probabilities based on current_state vector.
    current_state_vector hold states of all nodes
    Use np.random.multinomial to identify index in x_node states
    Solution for random assignment:
    http://stackoverflow.com/questions/4437250/choose-list-variable-given-probability-of-each-variable
    '''
    
    ''' Initialize node states "beliefs" with node parameters (stored in matrix A)'''
    cur_probabilities = copy.deepcopy(A[x_node][x_node])
    ''' Go over all edges connected to the node'''
    for j_edge in range(len(A[0])):
        if j_edge!=x_node and A[x_node][j_edge] is not None:
            ''' Go over all states of node-x'''
            for i in range(len(node_states[x_node])):
                #print(cur_probabilities[i])
                #print(A)
                if x_node<j_edge:
                    cur_probabilities[i] = cur_probabilities[i]*A[x_node][j_edge][i][current_states[j_edge]]
                else:
                    cur_probabilities[i] = cur_probabilities[i]*A[x_node][j_edge][current_states[j_edge]][i]
    
    
    probabilities=normalize_message(cur_probabilities)
    
    ''' In case of multiple states choose one according to randomly generated number and state probability'''
    ind = np.where(np.random.multinomial(1,list(probabilities)))[0][0]
    #print("Probabilities and choice: ",list(probabilities),"Color: ",node_states[x_node][ind]," Current state: ",current_states," Current node: ",x_node) 
           
    return node_states[x_node][ind]


def calc_marginals(node_choices, states):
    marginals=[]
    for choices in node_choices:
        b = {}
        for item in choices:
            b[item] = b.get(item, 0) + 1
        margin=[]
        #print("Dictionary: ...",b)
        for state in states:
            margin.append(b.get(state,0)/len(choices))
            #print("Margin ", margin)
        marginals.append(margin)
    
    return marginals    

def get_random_assignment(number_of_nodes,states):
    '''
    Produce random assignments for node
    '''
    choices=[]
    number_of_states=len(states)
    for i in range(number_of_nodes):
        choices.append(states[randint(0, number_of_states-1)])
    
    return choices

def populate_matrix_with_params(J,graph_parameters, states):
    '''
    Place graph parameters into Graph matrix. Apply exp() function - since we used log() to calculate parameters. exp() is needed to bring back probability 
    For example, in case if 5 node tree (example below):
       - graph parameters vector will hold 26 elements: 5x2 (nodes x states) + 4x4 (edges x state combinations)
       - Graph matrix will be 4x4 matrix with 4 nodes and 4 edges
    
    '''
    g_index=0
    for i in range(len(J[0])):
        for j in range(len(J[0])):
            if i==j:
                for k in range(len(J[i][j])):
                    J[i][j][k] = math.exp(graph_parameters[g_index])
                    g_index+=1
            if i < j and J[i][j] is not None:
                for k in range(len(states)):
                    for l in range(len(states)):
                        J[i][j][k][l] = math.exp(graph_parameters[g_index])
                        g_index+=1
            if i > j and J[i][j] is not None:
                J[i][j] = J[j][i]
    
    #print(J)
    

def gibbs(A,graph_parameters, burnin, its,initial_assignment=None):
    '''
    A - graph matrix - re-populated with graph_paramters
    graph_parameters - are theta's for nodes and edges. Replacing message function
    burnin - number of samples to throw - system is waring-up
    its - number of samples to generate
    '''
    
    states=[0,1]
    populate_matrix_with_params(A,graph_parameters,states)
    print('Gibs matrix after populate:',A)
    node_states=[]
    number_of_nodes = len(A[0])
    out_choices = []
    for i in range(number_of_nodes):
        new=[]
        out_choices.append(new)
    
    for n in range(number_of_nodes):
        node_states.append(states)
    
    if initial_assignment is None:
        choices = get_random_assignment(number_of_nodes,states)
    else:
        choices=initial_assignment
        
    print("Gibbs initial choices are: ",choices, "Burn-in and Iterations: ",burnin,its)
    for i in range(burnin):
        #print(A)
        for node_j in range(number_of_nodes):
            next_choice = calculate_state_given_states(node_j,A,node_states,choices)
            #print("Current choices, node, new-choice ",choices,j,next_choice)
            choices[node_j] = next_choice
    
    print("Choices after burn-in", choices)
    for i in range(its):
        for node_j in range(number_of_nodes):
            next_choice = calculate_state_given_states(node_j,A,node_states,choices)
            #print("Current choices, node, new-choice ",choices,j,next_choice)
            out_choices[node_j].append(next_choice)
            choices[node_j] = next_choice
    
    marginals = calc_marginals(out_choices,states)
    #print ("Final choices:..",out_choices[0])
    #print ("Final Choices Length ", len(out_choices),len(out_choices[0]))
    #print("Marginals are: ",marginals)
    #return marginals
    return np.matrix(out_choices)


if __name__ == '__main__':
    '''******************************************************************************************************
       Start execution 
       ******************************************************************************************************
    '''
    RUN_1=True
    
    if RUN_1:
    
        print("")
        '''
                    0
                  /  \
                1      2
              /  \    
            3     4
        '''
        print("***********************************************")
        
        
        #print("# 5 Node tree (below)")
        J=[
            [[0.5,0.5],[[0.25,0.25],[0.25,0.25]],[[0.25,0.25],[0.25,0.25]],None,None],
           [[[0.25,0.25],[0.25,0.25]],[0.5,0.5],None,[[0.25,0.25],[0.25,0.25]],[[0.25,0.25],[0.25,0.25]]],
           [[[0.25,0.25],[0.25,0.25]],None,[0.5,0.5],None,None],
           [None,[[0.25,0.25],[0.25,0.25]],None,[0.5,0.5],None],
           [None,[[0.25,0.25],[0.25,0.25]],None,None,[0.5,0.5]]
           ]
    
        ''' {0,1} coloring (almost)'''
        graph_parameters = \
        [ 1,  1,#0
           1,  2,  2,  1, #0->1
          1,  2,  2,  2,  #0->2
          1,1,#1
          1,  2,  2,  1,#1->3
          1,  2,  2,  1,#1->4
          1,1, #2
          1,1, #3
          1,1] #4
        
        graph_parameters = \
        [ 1,  1,    
         1,  1,  2,  2, #0->1 
         1,  1,  2,  2, #0->2
         1,  1, #1    
         1,  1,  2,  2,  #1->3
         1,  1,  2,  2, #1->4
         1,  1, #2
         1,  1, #3
         1,  1] #4
    else:
    
        '''
                    0
                  /  \
                1      2
        '''
        
        J=[[[1,1],[[1,1],[1,1]],[[1,1],[1,1]]],
           [[[1,1],[1,1]],[1,1],None],
           [[[1,1],[1,1]],None,[1,1]]]
        #print('Graph Matrix before call to gibbs:',J)
        graph_parameters = [ 1,1,   1,2,2,1, 1,2,2,1,   1,1,  1,1]
        #graph_parameters=np.exp(graph_parameters)
    
    samples = gibbs(J,graph_parameters,2**4, 2**4)
    
    empirical_prob = np.array(learn_parameters_qpbo_optimization_v1.calc_emp_probabilities(samples,J))
    
    print('Gibs probablities using original graph params', empirical_prob)
    print('Graph Matrix before call Learning params:',J)
    print('Samples before calling Learn new params:')
    print(samples)
    '''
    Learning parameters of the models given graph structure and samples
    '''
    
    '''
    p_params = learn_parameters_qpbo_optimization.learn_parameters(J,samples,[0,1],10)
    
    new_samples = gibbs(J,p_params,2**16, 2**16)
    new_empirical_prob = np.array(learn_parameters_qpbo_optimization.calc_emp_probabilities(new_samples,J))
    
    print('Original graph parameters',graph_parameters)
    print('Learned parameters',p_params)
    print('Original Gibbs probabilities', empirical_prob)
    print('New Gibs probabilities', new_empirical_prob)
    
    
    divergence=0
    number_of_nodes=len(J[0])
    p_param_matrix = verify_model.rebuild_probabilities_matrix(J, p_params,[0,1])
    q_param_matrix = verify_model.rebuild_probabilities_matrix(J, graph_parameters,[0,1])
    print('P matrix:',p_param_matrix)
    print('Q matrix:',q_param_matrix)
    #generate nodes-states [[0,1],[0,1].....[0,1]]
    nodes_assignments = [[0,1]]*number_of_nodes
    #generate all possible assignment combinations for all nodes: 00000, 00001, 00011, 00111, 01111, ..... 2^num_nodes combinations 
    node_assign_combinations = list(itertools.product(*nodes_assignments))
    Z_p = verify_model.calculate_Z(p_param_matrix,node_assign_combinations)
    Z_q = verify_model.calculate_Z(q_param_matrix,node_assign_combinations)
    np_samples = np.array(samples).transpose() #Original samples
    for node_assign_comb in node_assign_combinations:
        p_prob=verify_model.calculate_assignment_prob(node_assign_comb,p_param_matrix,Z_p)
        q_prob=verify_model.calculate_assignment_prob(node_assign_comb,q_param_matrix,Z_q)
        e_count=np.where((np_samples == node_assign_comb).all(axis=1))
        e_prob=len(e_count[0])/np_samples.shape[0]
        print(node_assign_comb,'p_prob',p_prob,'q_prob',q_prob,' e_prob',e_prob)
        #TODO - fix q_prob calculation
        #divergence = divergence+p_prob*(math.log(p_prob/q_prob))
        divergence = divergence+p_prob*(math.log(p_prob/e_prob))
    print('Divegence: ', divergence)
    '''
    
    '''
    s
[[1, 0, 0, 1, 1], [1, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 1], [1, 1, 0, 1, 0]]
>>> np.matrix(s).transpose()
matrix([[1, 1, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 1, 1],
        [1, 0, 0, 1, 0]])
    '''