'''
Created on March 4, 2017

@author: Uri Smashnov

Goal: Find believes and joint believes using:
      - Frank-Wolfe algorithm: iterative first-order optimization for constraint convex optimization
      - Linear approximation and optimization
Graph Structure:
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
from scipy.constants.constants import alpha
from scipy.optimize import linprog

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
    '''
    Independent sets
    '''
    if (x+y) <= 1:
        return 1
    else:
        return 0

def message_function2(x,y):
    '''
    Graph coloring
    '''
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
    '''
    x=len(J_matrix[0])
    gradient_vector=[]
    J_flatten=[]
    Map_matrix=deepcopy(J_matrix)
    array_index=0
    for i in range(x):
        for j in range(i,x):
            Map_matrix[i][j]=[]
            #print(i,j,Map_matrix[i][j])
            if i==j:
                for t,state in enumerate(node_states[i]):
                    partial_grad = calc_x_log(phi_of_xi(state))+(calc_x_log(J_matrix[i][i][t]))*(neighbors_count[i]-1) 
                    gradient_vector.append(partial_grad)
                    J_flatten.append(J_matrix[i][j][t])
                    Map_matrix[i][j].append(array_index)
                    array_index+=1
            elif (J_matrix[i][j] is not None):
                for t,state_i in enumerate(node_states[i]):
                    for s,state_j in enumerate(node_states[j]):
                        partial_grad = calc_x_log(message_function(state_i,state_j)) - (calc_x_log(J_matrix[i][j][t][s])+1)
                        gradient_vector.append(partial_grad)
                        J_flatten.append(J_matrix[i][j][t][s])
                        Map_matrix[i][j].append(array_index)
                        array_index+=1
    #print('Flat matrix \n', len(J_flatten))
    #print(J_flatten)
    return gradient_vector,Map_matrix,J_flatten

def call_linear_optimizer(gradient_vector,A_inp,G_inp,b_inp,h_inp):
    res = linprog(gradient_vector, A_ub=G_inp, b_ub=h_inp,A_eq=A_inp, b_eq=b_inp)
    return res.x
    

def build_constraints_matrices(Map_matrix,dim,node_states):
    '''
    Return G,h,A,b parameters for linear optimization. Where constraints defined as:
    A*x = b
    G*x <= h
    '''
    G=[]
    A=[]
    b=[]
    h=[]
    
    x=len(Map_matrix[0])
    ''' First build constraints for single node distribution probability
    '''
    if DEBUG:
        print('Map matrix')
        print(Map_matrix)
        print("X= ",x)
        
    for i in range(x):
        matrix_record=[0]*dim
        for elememt in Map_matrix[i][i]:
            matrix_record[elememt]=1
        A.append(matrix_record)
        b.append(1)
    
    
    
    ''' Build i-->j constraints for joint distribution probability
    '''        
    for i in range(x):
        for j in range(i+1,x):
            if (Map_matrix[i][j] is not None) and (Map_matrix[i][j] !=[]):
                joint_prob=Map_matrix[i][j]
                #print("joint_prob i/j",i,j, joint_prob)
                for ni,node_i in enumerate(node_states[i]):
                    matrix_record=[0]*dim
                    for nj,node_j in enumerate(node_states[j]):
                        #print("ni,nj =", ni,ni+nj, joint_prob[ni*len(node_states[j])+nj])
                        matrix_record[joint_prob[ni*len(node_states[j])+nj]] = 1
                    matrix_record[Map_matrix[i][i][ni]] = -1
                    A.append(matrix_record)
                    b.append(0)
                    
    ''' Build j--> i constraints for joint distribution probability
    '''        
                    
    ind=0                
    for i in range(x):
        for j in range(i+1,x):
            if (Map_matrix[i][j] is not None) and (Map_matrix[i][j] !=[]):
                joint_prob=Map_matrix[i][j]
                #print("joint_prob ",joint_prob)
                for nj,node_j in enumerate(node_states[j]):
                    matrix_record=[0]*dim
                    for ni,node_i in enumerate(node_states[i]):
                        matrix_record[joint_prob[nj + len(node_states[i])*ni]] = 1
                    matrix_record[Map_matrix[j][j][nj]] = -1
                    #if ind<2:
                    #if ind%2 ==0:
                    A.append(matrix_record)
                    b.append(0)
                    #ind+=1
    
                    
    '''
    Build G and h - used for less than ( G*x <= h)  constraint
    '''
    for elem in range(dim):
        matrix_record=[0]*dim
        matrix_record[elem] = -1
        G.append(matrix_record)
        h.append(0)
    
    if DEBUG:
        print("Matrix A: ")
        print_m(A)
        print("Vector b= ",b)
        print("Matrix G: ")
        print_m(G)
        print("Vector h= ",h)
    return A,G,b,h

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

def gradient_optimizer(J_matrix,node_states,message_function1):
    '''
    Frank-Wolfe algorithm, using  Linear optimizer
    '''
    neighbors_count=calculate_neighbors(J_matrix)
    gradient_vector,Map_matrix,J_flatten=calculate_gradient_vector(J_matrix,node_states,message_function1,neighbors_count)
    if DEBUG:
        print('Initial gradient vector', gradient_vector)
    A,G,b,h = build_constraints_matrices(Map_matrix,len(gradient_vector),node_states)
    for t in range(3000):
        tao_t_vector = call_linear_optimizer(np.array(gradient_vector)*(-1),A,G,b,h)
        if DEBUG:
            print("Shapes..",len(tao_t_vector),np.array(J_flatten).shape)
        alpha_t=2/(2+t)
        tao_t_plus_one=(1-alpha_t)*np.array(J_flatten) + alpha_t*tao_t_vector
        #print(type(tao_t_plus_one),tao_t_plus_one[0],tao_t_plus_one)
        J_matrix=rebuild_probabilities_matrix(J_matrix,tao_t_plus_one,node_states)
        #print(J_matrix)
        gradient_vector,Map_matrix,J_flatten=calculate_gradient_vector(J_matrix,node_states,message_function1,neighbors_count)
        if DEBUG:
            print('Gradient vector for iteration ',t,'...', gradient_vector)
    
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
gradient_optimizer(J,node_states,message_function1)



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
gradient_optimizer(J,node_states,message_function1)    


'''
                0
              /  \
            1      2
          /  \    
        3     4

5000 IT
[ 0.71436985  0.28563015  0.57126767  0.14310218  0.28563015  0.
  0.35708498  0.35728486  0.28563015  0.          0.85689782  0.14310218
  0.42835433  0.42854349  0.14310218  0.          0.42835433  0.42854349
  0.14310218  0.          0.64271514  0.35728486  0.57145651  0.42854349
  0.57145651  0.42854349]
'''