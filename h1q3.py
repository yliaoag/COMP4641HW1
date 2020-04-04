import os
import numpy as np
import scipy
import collections
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import csgraph_from_dense
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import shortest_path
from collections import Counter
from itertools import product


def get_community(labels):
    communities = list()
    group = list()
    for i in range(len(labels)):
        if i == 0:
            group.append(i)
            continue
        if labels[i] == labels[i-1]:
            group.append(i)
        else:
            communities.append(group)
            group = list()
            group.append(i)
        if i == len(labels)-1:
            communities.append(group) 
    return communities  

def degree(adjacency, num_of_nodes):
    for i in range(num_of_nodes):
        for j in range(num_of_nodes):
            if adjacency[i][j] == np.inf:
                adjacency[i][j] = 0
    G_degree = np.sum(adjacency, axis=1)
    degree_dict = dict()
    for i in range(len(G_degree)):
        degree_dict.update({str(i):G_degree[i]})
    return degree_dict

def remove_edges(deleted_edges, adj):
    for i in range(len(deleted_edges)):
        adj[deleted_edges[i][0]][deleted_edges[i][1]] = np.inf
        adj[deleted_edges[i][1]][deleted_edges[i][0]] = np.inf
    return adj

def betweenness(graph_data, num_of_nodes):
    paths = list()
    edges_count = Counter()
    for j in range(num_of_nodes): 
        for i in range(num_of_nodes):
            tmp = j
            path = list()
            while(tmp != -9999):
                pre = tmp
                tmp = graph_data[tmp][i]
                if tmp == -9999:
                    break
                path.append(tuple(sorted((pre, tmp))))
            if sorted(path) not in paths:
                paths.append(sorted(path))
                edges_count.update(sorted(path))
    #edges_count
    return edges_count

def highest_betweeness(count):
    most_frequent = count.most_common(1)
    deleted_edges = list()
    #print(most_frequent[0][1])
    for key, value in dict(count).items():
        if value == most_frequent[0][1]:
            deleted_edges.append(key)
            del count[key]
    return deleted_edges

def network_decomposition(adjacent_matrix, num_of_nodes):
    Graph = csgraph_from_dense(adjacent_matrix, null_value=np.inf)
    G_shortest_path = shortest_path(Graph, directed = False, return_predecessors = True)
    predecessors = np.array(G_shortest_path[1])
    predecessors = np.transpose(predecessors)

    counter = betweenness(predecessors, num_of_nodes)
    #print(counter)
    removed_edges = highest_betweeness(counter)
    adj = remove_edges(removed_edges, adjacent_matrix)

    decomposed_graph = csgraph_from_dense(adjacent_matrix, null_value=np.inf)
    n_components, labels = connected_components(csgraph=decomposed_graph,   directed=False, return_labels=True)
    return adj, labels, counter 

def modularity(g_adj, g_edges, num_of_nodes, communities):
    out_degree = dict(degree(g_adj, num_of_nodes))
    in_degree = out_degree
    norm = 1 / (2*(len(g_edges)))
    q = 0.0
    for c in communities:
        for u,v in product(c, repeat=2):
            q = q + g_adj[u][v] - in_degree.get(str(u))*out_degree.get(str(v))*norm
    return q*norm

#read in the txt file
inputfile = open('./input.txt')

first_line = True
adjacent_matrix = list()
num_of_nodes = 0
for line in inputfile:
    line = line.strip()
    if (first_line):
        num_of_nodes = int(line)
        first_line = False
        continue
    tmp = list()
    for num in line:
        if num != ' ':
            num = int(num)
            tmp.append(num)
    adjacent_matrix.append(tmp)

#print(num_of_nodes)

graph_adj = adjacent_matrix
adjacent_matrix = np.array(adjacent_matrix, float)

graph_edges = list()
for i in range(num_of_nodes):
    for j in range(num_of_nodes):
        if adjacent_matrix[i][j] == 0:
            adjacent_matrix[i][j] = np.inf
        else:
            if tuple(sorted((str(i), str(j)))) not in graph_edges:
                graph_edges.append(tuple(sorted((str(i), str(j)))))
graph = dict({'edges':graph_edges, 'adjacency': graph_adj})

modularities = list()
communities = list()
while (True):
    adj, labels, edges_counter = network_decomposition(adjacent_matrix, num_of_nodes)
    #print(adj)
    #print(labels)
    #print(edges_counter)
    adjacent_matrix = adj
    community = get_community(labels)
    m = round(modularity(graph.get('adjacency'), graph.get('edges'), num_of_nodes, community), 4)
    #print(m)
    modularities.append(m)
    communities.append(tuple(community))
    if not edges_counter:
        break


print("network decomposition: ", end='\n\n')
for c in (communities):
    print(c)
for c in communities:
    print(len(c), end = '')
    print(" clusters: modularity", modularities[communities.index(c)])
print("optimal structure: ", end='')
print(communities[modularities.index(max(modularities))])