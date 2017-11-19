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

        
def binary_graph(np.ndarray[np.int32_t, ndim=2, mode='c'] edges,
                 np.ndarray[np.int32_t, ndim=2, mode='c'] unary_cost,
                 np.ndarray[np.int32_t, ndim=2, mode='c'] pairwise_cost):
    """QPBO inference on a graph with binary variables.
    
    Pairwise potentials are the same for all edges.

    Parameters
    ----------
    edges : nd-array, shape=(n_edges, 2)
        Edge-list describing the graph. Edges are given
        using node-indices from 0 to n_nodes-1.

    unary_cost : nd-array, shape=(n_nodes, 2)
        Unary potential costs. Rows correspond to rows, columns
        to states.
    
    pairwise_cost : nd-array, shape=(2, 2)
        Symmetric pairwise potential.

    Returns
    -------
    result : nd-array, shape=(n_nodes,)
        Approximate MAP as inferred by QPBO.
        Values are 0, 1 and -1 for non-assigned nodes.

    """
    cdef int n_nodes = unary_cost.shape[0]
    if unary_cost.shape[1] != 2:
        raise ValueError("unary_cost must be of shape (n_nodes, 2).")
    if edges.shape[1] != 2:
        raise ValueError("edges must be of shape (n_edges, 2).")
    if pairwise_cost.shape[0] != pairwise_cost.shape[1]:
        raise ValueError("pairwise_cost must be square matrix.")
    if (pairwise_cost != pairwise_cost.T).any():
        raise ValueError("pairwise_cost must be symmetric.")
    cdef int n_edges = edges.shape[0] 
    # create qpbo object
    cdef QPBO[int] * q = new QPBO[int](n_nodes, n_edges)
    q.AddNode(n_nodes)
    cdef int* data_ptr = <int*> unary_cost.data
    # add unary terms
    for i in xrange(n_nodes):
        q.AddUnaryTerm(i, data_ptr[2 * i], data_ptr[2 * i + 1])
    # add pairwise terms
    # we have global terms
    cdef int e00 = pairwise_cost[0, 0]
    cdef int e10 = pairwise_cost[1, 0]
    cdef int e01 = pairwise_cost[0, 1]
    cdef int e11 = pairwise_cost[1, 1]

    for e in edges:
        q.AddPairwiseTerm(e[0], e[1], e00, e10, e01, e11)

    q.Solve()
    q.ComputeWeakPersistencies()

    # get result
    cdef np.npy_intp result_shape[1]
    result_shape[0] = n_nodes
    cdef np.ndarray[np.int32_t, ndim=1] result = np.PyArray_SimpleNew(1, result_shape, np.NPY_INT32)
    cdef int * result_ptr = <int*>result.data
    for i in xrange(n_nodes):
        result_ptr[i] = q.GetLabel(i)

    del q
    return result


def alpha_expansion_general_graph(
        np.ndarray[np.int32_t, ndim=2, mode='c'] edges,
        np.ndarray[np.int32_t, ndim=2, mode='c'] unary_cost,
        np.ndarray[np.int32_t, ndim=3, mode='c'] edge_costs,
        int n_iter=5, verbose=False, random_seed=None):
    """Alpha expansion using QPBO inference on general graph.
    
    Pairwise potentials can be arbitrary, given by edge_costs.
    For the i'th edge in ``edges`` the cost of connecting labels
    y_1 and y_2 is given by ``edge_costs[i, y_1, y_2]``.

    Alpha expansion is very efficient but inference is only approximate and
    none of the persistence properties of QPBO are preserved.

    Parameters
    ----------
    edges : nd-array, shape=(n_edges, 2)
        Edge-list describing the graph. Edges are given
        using node-indices from 0 to n_nodes-1.

    unary_cost : nd-array, shape=(n_nodes, n_states)
        Unary potential costs.

    edges_cost : nd-array, shape=(n_edges, n_states, n_states)
        Symmetric pairwise potential.

    n_iter : int, default=5
        Number of expansion iterations (how often to go over labels).

    verbose : int, default=0
        Verbosity.

    random_seed: int or None
        If int, a fixed random seed is used for reproducable results.

    Returns
    -------
    result : nd-array, shape=(n_nodes,)
        Approximate MAP as inferred by QPBO.
        Values are 0, 1 and -1 for non-assigned nodes.

    """

    cdef int n_nodes = unary_cost.shape[0]
    cdef int n_labels =  unary_cost.shape[1]
    cdef int n_edges = edges.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1] x
    cdef int old_label
    cdef int label
    cdef int changes
    cdef int e00, e01, e10, e11
    cdef int edge0, edge1

    if random_seed is None:
        rnd_state = np.random.mtrand.RandomState()
        srand(time())
    else:
        rnd_state = np.random.mtrand.RandomState(random_seed)
        srand(random_seed)

    # initial guess
    x = np.zeros(n_nodes, dtype=np.int32)
    cdef int* edge_ptr = <int*> edges.data
    cdef int* x_ptr = <int*> x.data
    cdef int* x_ptr_current

    cdef int* data_ptr = <int*> unary_cost.data
    cdef int* data_ptr_current

    # create qpbo object
    cdef QPBO[int] * q = new QPBO[int](n_nodes, n_edges)
    #cdef int* data_ptr = <int*> unary_cost.data
    for n in xrange(n_iter):
        if verbose > 0:
            print("iteration: %d" % n)
        changes = 0
        for alpha in rnd_state.permutation(n_labels):
            q.AddNode(n_nodes)
            for i in xrange(n_nodes):
                # first state is "keep x", second is "switch to alpha"
                # TODO: what if state is already alpha? Need to collapse?
                if alpha == x[i]:
                    q.AddUnaryTerm(i, unary_cost[i, x_ptr[i]], 100000)
                else:
                    q.AddUnaryTerm(i, unary_cost[i, x_ptr[i]], unary_cost[i, alpha])
            for e in xrange(n_edges):
                edge0 = edge_ptr[2 * e]
                edge1 = edge_ptr[2 * e + 1]
                #down
                e00 = edge_costs[e, x_ptr[edge0], x_ptr[edge1]]
                e01 = edge_costs[e, x_ptr[edge0], alpha]
                e10 = edge_costs[e, alpha, x_ptr[edge1]]
                e11 = edge_costs[e, alpha, alpha]
                q.AddPairwiseTerm(edge0, edge1, e00, e01, e10, e11)

            q.Solve()
            q.ComputeWeakPersistencies()
            improve = True
            while improve:
                improve = q.Improve()

            for i in xrange(n_nodes):
                old_label = x_ptr[i]
                label = q.GetLabel(i)
                if label == 1:
                    x_ptr[i] = alpha
                    changes += 1
                if label < 0:
                    print("LABEL <0 !!!")
            # compute energy:
            q.Reset()
        if verbose > 0:
            print("changes: %d" % changes)
        if changes == 0:
            break
    del q
    return x


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
