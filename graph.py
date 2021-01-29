import numpy as np
import itertools 

class Graph:
    def __init__(self, N, weight = 1):
        self.N = N
        self.init_nodes()
        self.edges = []
        self.isolated_nodes = []
        self.list_edges = []
        self.weight = weight
        
    def init_nodes(self):
        self.graph = {int_val: dict() for int_val in range(1, self.N + 1)}
        
    def nodes(self):
        return list(self.graph.keys())
    
    def number_of_vertices(self):
        return len(self.edges)
    
    def get_vertices(self, node):
        return list(self.graph[node].keys())
    
    def find_edge(self, node1, node2):
        try:
            return self.graph[node1][node2]
        except KeyError:
            return None
    
    def add_edge(self, node1, node2):
        u,v = sorted([node1, node2])
        if (u-1, v-1) not in self.edges and u != v:
            self.graph[u][v] = self.weight
            self.graph[v][u] = self.weight
            self.edges.append((u-1, v-1))
    
    def get_knn(self, padded_list, j, m):
        return sorted(padded_list[j - m : j][:: -1] + padded_list[j + 1: j + m + 1])
    
    def add_nn_edges(self, k, weight):
        m = int(k/2)
        #Set lattice vertices
        padded_list = self.nodes() * 3
        self.graph = {self.nodes()[i]: {w: self.weight for w in self.get_knn(padded_list, j, m)}
                      for i, j in zip(range(self.N), range(self.N, 2 * self.N))}
        nodes = self.nodes()
        self.list_edges = [l for sl in 
                           [[(u,v) for u,v in zip(nodes, nodes[j:] + nodes[0:j])] for j in range(1, k // 2+1)] 
                           for l in sl]
        
        for e in self.list_edges: 
            u ,v = e
            self.edges.append(tuple(sorted([u-1, v-1])))
                                        
    def find_isolated_nodes(self):
        if any(edges != {} for edges in self.graph.values()):
            pass
        else:
            for node, edges in self.graph.items():
                if edges == {}: self.isolated_nodes.append(node)
    
    def remove_edge(self, node1, node2):
        u, v = sorted([node1, node2])
        if self.find_edge(u, v) != None:
            del self.graph[u][v]
            del self.graph[v][u]
            self.edges.remove((u - 1, v - 1))
               
    def adjacency_matrix(self):
        A = np.zeros((self.N, self.N))
        for i in self.edges: 
            r, c = i
            A[r][c] = self.weight
            A[c][r] = A[r][c]
        
        return A
   
    def is_connected(self):
        self.find_isolated_nodes()
        if any(self.isolated_nodes):
            return False
        
        v0 = self.nodes()[0]
        queue = []
        visited = []
        queue.append(v0)
        
        while (queue):
            v = queue[0]
            visited.append(v)
            queue.remove(queue[0])
            for node in self.graph[v]:
                if node not in visited and node not in queue:
                    queue.append(node)

        if set(visited) == set(self.nodes()):
            return True
        else:
            return False
