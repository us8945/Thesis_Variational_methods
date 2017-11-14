'''
Created on Feb 19, 2017

@author: UriS

-- Add Z calculation using Beth energy approximation

Basic Belief Propagation example.
Flow:
    - Define Matrix J such that:
        - [i][i] element of matrix contains marginal probability for element i. Initialized proportionally to number of states (0.5 for two states model)
        - [i][j] element of matrix contains message from i to j (i->j). Initialized to [1,1...1]. Each message has same number of elements as states in node j
        - [i][j] is None in case node i is not connected to node j
    - Define vector H to hold weight for each node. Number of elements as number of nodes
    - Define array States to hold list of states (colors) for each node
Flow:
1) Run sum-prod BP algorithm on input matrix, color weights and number of iterations
2) Use sum-prod elimination to come up with beliefs for nodes assignment of 0 or 1
3) Output partition function Z


TODO:
Weights are currently not in use and coded to be weights on node state. Adapted from graph coloring.


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
print("")
print("Z = ",sumprod(A,weights,node_states,message_function1,50))
'''
import numpy as np
import math

def calc_x_log_y(x,y):
    if (x==0 or y==0):
        return 0
    else:
        return x*math.log(y)

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
    '''
    Independent sets function
    '''
    if (x+y) <= 1:
        return 1
    else:
        return 0

def message_function2(x,y):
    '''
    Graph coloring function
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
            if mem is None:
                l=l+["None"]
            else:
                l=l+[[prettyfloat(n) for n in mem]]
        print(l)
    #print(np.array(J_matrix))
    
        
def pass_over_matrix(col_weigths,J_matrix,node_states,message_function,iterations=50):
    x=len(J_matrix[0])
    for k in range(iterations):
        for i in range(x):
            for j in range(x):
                message_to_i = incoming_messages(i,len(node_states[i]),j,J_matrix) #all incoming message exclude message from j
                if i!=j and J_matrix[i][j] is not None: #send message from i to j
                    #print('X,I and J: ',x,i,j)
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


'''******************************************************************************************************
   Start execution 
   ******************************************************************************************************
'''
print("")
print("***********************************************")
print("# Three node loop graph, weigths=[1,1,1]")
print('Independent sets problem')
A=[[[0.5,0.5],[1,1],[1,1]],
   [[1,1],[0.5,0.5],[1,1]],
   [[1,1],[1,1],[0.5,0.5]]]
weights=[1,1]
node_states=[[0,1],[0,1],[0,1]]
print("")
print("Z = ",sumprod(A,weights,node_states,message_function1,50))    


print("")
print("***********************************************")
print("# 5 Node tree (below), weights=[1,1,1]")
node_states=[[0,1],[0,1],[0,1],[0,1],[0,1]]
weights=[1,1]
A=[[[0.5,0.5],[1,1],[1,1],None,None],
   [[1,1],[0.5,0.5],None,[1,1],[1,1]],
   [[1,1],None,[0.5,0.5],None,None],
   [None,[1,1],None,[0.5,0.5],None],
   [None,[1,1],None,None,[0.5,0.5]]]
print("")
print("Z = ",sumprod(A,weights,node_states,message_function1,50))    

'''
                0
              /  \
            1      2
          /  \    
        3     4

'''

print("")
print("***********************************************")
print("# 4 node loop graph, weigths=[1,1,1,1]")
print('Graph coloring problem')
A=[[[0.5,0.5,0.5],[1,1,1],None,[1,1,1]],
   [[1,1,1],[0.5,0.5],[1,1,1],None],
   [None,[1,1,1],[0.5,0.5,0.5],[1,1,1]],
   [[1,1,1],None,[1,1,1],[0.5,0.5,0.5]]]
weights=[1,1,1]
node_states=[[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
print("")
print("Z = ",sumprod(A,weights,node_states,message_function2,500))    