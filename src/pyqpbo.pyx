###############################################################
# Python bindings for QPBO algorithm by Vladimir Kolmogorov.
#
# Author: Uri Smashnov usmashnov@yahoo.com
# License: MIT
# Compile instructions:
# cython --cplus pyqpbo.pyx

#To build package:
#- Open Native 64 bit prompt from C:\Program Files (x86)\Microsoft Visual C++ Build Tools
#- python setup.py build_ext --inplace
##################################################

import numpy as np
from numpy import matrix
cimport numpy as np
from libcpp cimport bool
from time import time

np.import_array()

cdef extern from "stdlib.h":
    void srand(unsigned int seed) 

ctypedef int NodeId
ctypedef int EdgeId


cdef extern from "QPBO.h":
    cdef cppclass QPBO[REAL]:
        QPBO(int node_num_max, int edge_num_max)
        bool Save(char* filename, int format=0)
        bool Load(char* filename)
        void Reset()
        NodeId AddNode(int num)
        void AddUnaryTerm(NodeId i, REAL E0, REAL E1)
        EdgeId AddPairwiseTerm(NodeId i, NodeId j, REAL E00, REAL E01, REAL E10, REAL E11)
        void AddPairwiseTerm(EdgeId e, NodeId i, NodeId j, REAL E00, REAL E01, REAL E10, REAL E11)
        int GetLabel(NodeId i)
        void Solve()
        void ComputeWeakPersistencies()
        void Improve()

        



def qpbo_solver_general_matrix(
        np.ndarray[np.float64_t, ndim=2, mode='c'] matrix,
        int n_nodes, int vector_length):
        #np.ndarray[np.float32_t, ndim=2, mode='c'] unary_cost, # Node cost - two values in case of {0,1}
        #np.ndarray[np.float32_t, ndim=3, mode='c'] edge_costs, # Edge cost - four values in case of {0,1}
        #int n_iter=5, verbose=False, random_seed=None):
    """
    Input Array representing upper diagonal part of the Graph matrix
    Each node is represented by two values corresponding to state {0,1}.
    Every edge is represented by four values corresponding to states {00,01,10,11}
    To use numpy array structure, node elements are 4 members long as well as edges
    """
    print("\n*****************************\n")
    
    #print('Matrix array',matrix)
    #print('Number of Nodes:',n_nodes)
    
    cdef QPBO[float] * q = new QPBO[float](n_nodes, vector_length - n_nodes)
    
    '''
    Very important to call AddNode. Otherwise QPBO will not work properly and no error message will be given
    '''
    q.AddNode(n_nodes)
    cdef int matrix_record_start = 0
    cdef int node_step = n_nodes
    cdef int edge0, edge1
    cdef int matrix_current_rec = 0
    labels=[]
    for i in range(vector_length):
        '''
        Add node
        '''
        if i==matrix_record_start:
            edge0 = matrix_current_rec
            edge1 = matrix_current_rec+1
            matrix_record_start+= node_step
            node_step-= 1
            #print('Matrix node ',matrix_current_rec,matrix[i])
            print('Add node: ',matrix_current_rec, matrix[i][0], matrix[i][1])
            q.AddUnaryTerm(matrix_current_rec, matrix[i][0], matrix[i][1])
            matrix_current_rec+=1
        else:
            '''
        Add edge
        '''
            #print('Matrix edge ',i,matrix[i])
            if np.count_nonzero(matrix[i])>0:
                print('Add edges: ', edge0, edge1, matrix[i][0], matrix[i][1], matrix[i][2], matrix[i][3])
                q.AddPairwiseTerm(edge0, edge1, matrix[i][0], matrix[i][1], matrix[i][2], matrix[i][3])
            edge1+=1
    
    q.Solve()
    for node in range(n_nodes):
        labels.append(q.GetLabel(node))
    
    #print('labels assignment: ', labels)
    del q

    '''
    cdef QPBO[float] * qq = new QPBO[float](4, 4)
    qq.AddNode(4)
    qq.AddUnaryTerm(0, 10.1, 1.1)
    qq.AddUnaryTerm(1, 1.1, 1.1)
    qq.AddUnaryTerm(2, 1.1, 1.1)
    qq.AddUnaryTerm(3, 1.1, 1.1)
    qq.AddPairwiseTerm(0, 1, 0.0, -1.1, -1.1, 0)
    qq.AddPairwiseTerm(1, 2, 0, -1, -1, 0)
    qq.AddPairwiseTerm(2, 3, 0, -1, -1, 0)
    qq.AddPairwiseTerm(0, 3, 0, -1, -1, 0)
    qq.Solve()
    new_labels=[]
    cdef int result
    for node in range(4):
        new_labels.append(qq.GetLabel(node))
    
    print('labels assignment: ', new_labels)
    del qq  
    '''
    return labels

def qpbo_solver_general_matrix_w_struct(
        np.ndarray[np.float64_t, ndim=2, mode='c'] matrix,
        graph_matrix,
        int n_nodes, int n_edges):
        #np.ndarray[np.float32_t, ndim=2, mode='c'] unary_cost, # Node cost - two values in case of {0,1}
        #np.ndarray[np.float32_t, ndim=3, mode='c'] edge_costs, # Edge cost - four values in case of {0,1}
        #int n_iter=5, verbose=False, random_seed=None):
    """
    Input Array representing upper diagonal part of the Graph matrix
    Each node is represented by two values corresponding to state {0,1}.
    Every edge is represented by four values corresponding to states {00,01,10,11}
    To use numpy array structure, node elements are 4 members long as well as edges
    """
    #print("\n*****************************\n")
    
    #print('Matrix array',matrix)
    #print('Number of Nodes:',n_nodes)
    #print('Graph matrix', graph_matrix)
    #print('Graph matrix type', type(graph_matrix))
    
    cdef QPBO[float] * q = new QPBO[float](n_nodes, n_edges)
    
    '''*****************************************************************************************************
    Very important to call AddNode. Otherwise QPBO will not work properly and no error message will be given
    '''
    q.AddNode(n_nodes)
    '''******************************************************************************************************
    '''
    cdef int matrix_current_rec = 0
    cdef int graph_current_rec = 0
    
    labels=[]
    label=0.0
    
    for i in range(n_nodes):
        for j in range(i,n_nodes):
            if i==j:
                '''
                Add node
                '''    
                #print ('Adding node from item:',matrix_current_rec,'Node i:',i,' Node j: ',j)
                q.AddUnaryTerm(i, matrix[matrix_current_rec][0], matrix[matrix_current_rec][1])
                matrix_current_rec+=1
            else:
                if graph_matrix[i][j] is not None:
                    '''
                    Add edge
                    '''
                    #print('Adding edge from item',matrix_current_rec,'Node i:',i,' Node j: ',j)
                    q.AddPairwiseTerm(i, j, matrix[matrix_current_rec][0], matrix[matrix_current_rec][1], matrix[matrix_current_rec][2], matrix[matrix_current_rec][3])
                    matrix_current_rec+=1    
            graph_current_rec+=1
        graph_current_rec+=1
            
    q.Solve()
    for node in range(n_nodes):
        label=q.GetLabel(node)
        if label<0:
            label=0.5
        labels.append(label)
        
    
    #print('labels assignment: ', labels)
    del q
    
    return labels

def qpbo_solver_general_matrix_w_struct_v2(
        np.ndarray[np.float64_t, ndim=2, mode='c'] matrix,
        graph_matrix,
        int n_nodes, int n_edges,
        np.ndarray[np.int32_t, ndim=1, mode='c'] sample,
        np.ndarray[np.int32_t, ndim=1, mode='c'] calc_array):
        #np.ndarray[np.float32_t, ndim=2, mode='c'] unary_cost, # Node cost - two values in case of {0,1}
        #np.ndarray[np.float32_t, ndim=3, mode='c'] edge_costs, # Edge cost - four values in case of {0,1}
        #int n_iter=5, verbose=False, random_seed=None):
    """
    Input Array representing upper diagonal part of the Graph matrix
    Each node is represented by two values corresponding to state {0,1}.
    Every edge is represented by four values corresponding to states {00,01,10,11}
    To use numpy array structure, node elements are 4 members long as well as edges
    """
    #print("\n*****************************\n")
    
    #print('Matrix array',matrix)
    #print('Number of Nodes:',n_nodes)
    #print('Graph matrix', graph_matrix)
    #print('Graph matrix type', type(graph_matrix))
    
    cdef QPBO[float] * q = new QPBO[float](n_nodes, n_edges)
    
    '''*****************************************************************************************************
    Very important to call AddNode. Otherwise QPBO will not work properly and no error message will be given
    '''
    q.AddNode(n_nodes)
    '''******************************************************************************************************
    '''
    cdef int matrix_current_rec = 0
    cdef int graph_current_rec = 0
    
    labels=[]
    label=0.0
    
    for i in range(n_nodes):
        for j in range(i,n_nodes):
            if i==j:
                '''
                Add node
                '''    
                if i in calc_array:
                    q.AddUnaryTerm(i, matrix[matrix_current_rec][0], matrix[matrix_current_rec][1])
                else:
                    if sample[i]==0:
                        q.AddUnaryTerm(i, -1, 100) #force node value to 0
                    else:
                        q.AddUnaryTerm(i, 100, -1) #force node value to 1
                matrix_current_rec+=1
            else:
                if graph_matrix[i][j] is not None:
                    '''
                    Add edge
                    '''
                    #print('Adding edge from item',matrix_current_rec,'Node i:',i,' Node j: ',j)
                    q.AddPairwiseTerm(i, j, matrix[matrix_current_rec][0], matrix[matrix_current_rec][1], matrix[matrix_current_rec][2], matrix[matrix_current_rec][3])
                    matrix_current_rec+=1    
            graph_current_rec+=1
        graph_current_rec+=1
            
    q.Solve()
    for node in range(n_nodes):
        label=q.GetLabel(node)
        if label<0:
            label=0.5
        labels.append(label)
        
    
    #print('labels assignment: ', labels)
    del q
    
    return labels

def qpbo_solver_general_matrix_w_struct1(
        np.ndarray[np.float64_t, ndim=2, mode='c'] matrix,
        graph_matrix,
        int n_nodes, int n_edges):
        #np.ndarray[np.float32_t, ndim=2, mode='c'] unary_cost, # Node cost - two values in case of {0,1}
        #np.ndarray[np.float32_t, ndim=3, mode='c'] edge_costs, # Edge cost - four values in case of {0,1}
        #int n_iter=5, verbose=False, random_seed=None):
    """
    Input Array representing upper diagonal part of the Graph matrix
    Each node is represented by two values corresponding to state {0,1}.
    Every edge is represented by four values corresponding to states {00,01,10,11}
    To use numpy array structure, node elements are 4 members long as well as edges
    """
    #print("\n*****************************\n")
    
    #print('Matrix array',matrix)
    #print('Number of Nodes:',n_nodes)
    #print('Graph matrix', graph_matrix)
    #print('Graph matrix type', type(graph_matrix))
    
    cdef QPBO[float] * q = new QPBO[float](n_nodes, n_edges)
    
    '''*****************************************************************************************************
    Very important to call AddNode. Otherwise QPBO will not work properly and no error message will be given
    '''
    q.AddNode(n_nodes)
    '''******************************************************************************************************
    '''
    cdef int matrix_current_rec = 0
    cdef int graph_current_rec = 0
    
    labels=[]
    label=0.0
    
    for i in range(n_nodes):
        for j in range(i,n_nodes):
            if i==j:
                '''
                Add node
                '''    
                #print ('Adding node from item:',matrix_current_rec,'Node i:',i,' Node j: ',j)
                q.AddUnaryTerm(i, matrix[matrix_current_rec][0], matrix[matrix_current_rec][1])
                matrix_current_rec+=1
            else:
                if graph_matrix[i][j] is not None:
                    '''
                    Add edge
                    '''
                    #print('Adding edge from item',matrix_current_rec,'Node i:',i,' Node j: ',j)
                    q.AddPairwiseTerm(i, j, matrix[matrix_current_rec][0], matrix[matrix_current_rec][1], matrix[matrix_current_rec][2], matrix[matrix_current_rec][3])
                    matrix_current_rec+=1    
            graph_current_rec+=1
        graph_current_rec+=1
            
    q.Solve()
    for node in range(n_nodes):
        label=q.GetLabel(node)
        if label<0:
            label=0.5
        labels.append(label)
    
    #print('labels assignment: ', labels)
    del q
    
    return labels
