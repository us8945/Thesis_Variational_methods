'''
Created on Sep 4, 2017

@author: Uri Smashnov

Purpose: validate training results

Steps:
1) Set up tree structure using matrix
2) Generate/populate parameters to be used as baseline - this is our "q" model
3) Generate samples using Gibbs sampling and parameters from "q" model
4) Use samples from #3 to reain "p" model
5) Compute D(p||q) = Sum p(x)*log((p(x)/q(x)) , where x is one of the possible combinations of nodes assignment of the tree(graph).
   For example, 5 node tree below would have 2^5 possible assignments (assuming {0,1} assignments).
       0
      /\
     1  2
   /  \
 3    4
 
6) D(p||q) need to go to zero as sampling size increases
'''

from optimization import gibs_sampling, learn_parameters_qpbo_optimization_v1
import numpy as np
import itertools
import math
from copy import deepcopy

def rebuild_probabilities_matrix(J_m, probabilities_arr,node_states):
    #print(J_matrix,"\n",probabilities_arr,"\n",node_states)
    J_new = deepcopy(J_m)
    x=len(J_new[0])
    array_index=0
    for i in range(x):
        for j in range(i,x):
            if i==j:
                for t,state in enumerate(node_states):
                    J_new[i][j][t] = probabilities_arr[array_index]
                    array_index+=1
            elif (J_new[i][j] is not None):
                for t,state_i in enumerate(node_states):
                    for s,state_j in enumerate(node_states):
                        J_new[i][j][t][s] = probabilities_arr[array_index]
                        array_index+=1
    return J_new

 
def calculate_samples_prob(node_assignment,samples):
    '''
    Calculate probability of particular assignment
    '''
    np_samples = np.array(samples).transpose()
    number_of_samples=len(np_samples)
    emp_probabilities=[]
    
    
    
def calculate_assignment_prob(node_assignment,param_matrix,Z):
    '''
    Calculate probability of particular assignment
    '''
    prob=1
    #print('Calculate Assignement probability. Node assignment: ',node_assignment)
    #print('Param matrix: ',param_matrix)
    for i in range(len(param_matrix[0])):
        for j in range(i,len(param_matrix[0])):
            if i==j:
                prob=prob*math.exp(param_matrix[i][j][node_assignment[i]])
                #print('param ',param_matrix[i][j][node_assignment[i]],'i:',i,'j: ',j,'current prob:',prob)
            elif param_matrix[i][j] is not None:
                prob=prob*math.exp(param_matrix[i][j][node_assignment[i]][node_assignment[j]])
                
                #print('param ',param_matrix[i][j][node_assignment[i]][node_assignment[j]],'i:',i,'j: ',j,'current prob:',prob)
    #print('Prob before Z',prob)
    return prob/Z

def calculate_Z(param_matrix,node_assign_combinations):
    '''
    Indexing for Edge:
    - Edge has four parameters for the following states combinations:[[00,01],[10,11]].
    - Makes easy to access using state on n_i first and n_j second
    '''
    Z=0.0
    #print(param_matrix)
    #print(node_assign_combinations)
    for assignment in node_assign_combinations:
        z=1
        for i in range(len(param_matrix[0])):
            for j in range(i,len(param_matrix[0])):
                if i==j:
                    z=z*math.exp(param_matrix[i][i][assignment[i]])
                elif param_matrix[i][j] is not None:
                    #print(i,j,assignment[i],assignment[j])
                    #print('Edge: ',param_matrix[i][j])
                    z=z*math.exp(param_matrix[i][j][assignment[i]][assignment[j]])
        #print('Combination:',assignment,' probability:',z)
        Z=Z+z
    print ('Z:',Z)
    return Z

def calc_divergence(p_parameters,q_parameters,J_matrix,node_states,samples,debug_ind=False):
    '''
    Calculate The Kullback–Leibler divergence from Q to P is often denoted Dkl(P‖Q).
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    '''
    divergence=0
    number_of_nodes=len(J_matrix[0])
    p_param_matrix = rebuild_probabilities_matrix(J_matrix, p_parameters,node_states)
    q_param_matrix = rebuild_probabilities_matrix(J_matrix, q_parameters,node_states)
    
    '''generate nodes-states [[0,1],[0,1].....[0,1]]'''
    nodes_assignments = [[0,1]]*number_of_nodes
    ''' generate all possible assignment combinations for all nodes: 00000, 00001, 00011, 00111, 01111, ..... 2^num_nodes combinations''' 
    node_assign_combinations = list(itertools.product(*nodes_assignments))
    Z_p = calculate_Z(p_param_matrix,node_assign_combinations)
    Z_q = calculate_Z(q_param_matrix,node_assign_combinations)
    np_samples = np.array(samples).transpose() #Original samples
    for node_assign_comb in node_assign_combinations:
        p_prob=calculate_assignment_prob(node_assign_comb,p_param_matrix,Z_p)
        q_prob=calculate_assignment_prob(node_assign_comb,q_param_matrix,Z_q)
        e_count=np.where((np_samples == node_assign_comb).all(axis=1))
        e_prob=len(e_count[0])/np_samples.shape[0]
        if debug_ind:
            print(node_assign_comb,'p_prob',p_prob,'q_prob',q_prob,' e_prob',e_prob)
        divergence = divergence+p_prob*(math.log(p_prob/q_prob))
    
    return divergence

if __name__ == '__main__':
    '''******************************************************************************************************
       Start execution 
       ******************************************************************************************************
    '''
    
    print("")
    print("Use Tree below")
    print("            0") 
    print("        '/'  '\\'")    
    print("        1      2")
    print("    '/' '\\'")
    print("    3     4")
    
    print("***********************************************")
    print("# 5 Node tree (above)")
    J=[
        [[0.5,0.5],[[0.25,0.25],[0.25,0.25]],[[0.25,0.25],[0.25,0.25]],None,None],
       [[[0.25,0.25],[0.25,0.25]],[0.5,0.5],None,[[0.25,0.25],[0.25,0.25]],[[0.25,0.25],[0.25,0.25]]],
       [[[0.25,0.25],[0.25,0.25]],None,[0.5,0.5],None,None],
       [None,[[0.25,0.25],[0.25,0.25]],None,[0.5,0.5],None],
       [None,[[0.25,0.25],[0.25,0.25]],None,None,[0.5,0.5]]
       ]

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
    edges_weight=[2,2,2,2]
    
    n_samples=2**8
    samples = gibs_sampling.gibbs(J,graph_parameters,n_samples, n_samples)
    print('Samples matrix',samples.shape,samples)
    node_states=[0,1]
    regularization=1
    itr=50
    
    
    #p_params = learn_parameters_qpbo_optimization_v0.learn_parameters(J,samples,node_states,50)
    p_params = learn_parameters_qpbo_optimization_v1.learn_parameters(J,samples,node_states,edges_weight,regularization, itr)
    print('Original parameters: ',graph_parameters)
    print('Learned params:', p_params)
    print('Number of samples: ',n_samples)
    divergence= calc_divergence(p_params,graph_parameters,J,node_states,samples,False)
    print('Divergence: ',divergence)
    
    print("")
    print("Use Tree below")
    print("              0")
    print("              |")
    print("              1") 
    print("        '/'     '\\'")    
    print("        2          3")

    print("Node 1 is hidden and Node zero is label node")
    
    print("***********************************************")
    print("# 5 Node tree (above)")
    J=[
        [[0.5,0.5],[[0.25,0.25],[0.25,0.25]],None,None],
       [[[0.25,0.25],[0.25,0.25]],[0.5,0.5],[[0.25,0.25],[0.25,0.25]],[[0.25,0.25],[0.25,0.25]]],
       [None, [[0.25,0.25],[0.25,0.25]],[0.5,0.5],None],
       [None, [[0.25,0.25],[0.25,0.25]],None,[0.5,0.5]]
       ]
    
    hidden_arr=[0]
    label_arr=[1]
    graph_parameters = \
        [ 1,  1,    
         0,  0,  0,  2, #0->1 if n1=1 => n0=1 
         1,  1, #1    
         1,  1,  1,  1,  #1->2
         1,  1,  1,  1, #1->3
         1,  1, #2
         2,  0] #3
         
    edges_weight=[1,1,1]
    
    n_samples=2**8
    samples = gibs_sampling.gibbs(J,graph_parameters,n_samples, n_samples)
    print('Probabilities:',np.array(learn_parameters_qpbo_optimization_v1.calc_emp_probabilities(samples,J)))
    print('Samples matrix',samples.shape,samples)
    node_states=[0,1]
    regularization=.15
    itr=500
    
    
    #p_params = learn_parameters_qpbo_optimization_v0.learn_parameters(J,samples,node_states,50)
    p_params = learn_parameters_qpbo_optimization_v1.learn_parameters(J,samples,node_states,edges_weight,regularization, itr)
    print('Original parameters: ',graph_parameters)
    print('Learned params:', p_params)
    print('Number of samples: ',n_samples)
    divergence= calc_divergence(p_params,graph_parameters,J,node_states,samples,True)
    print('Divergence: ',divergence)


