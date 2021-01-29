from graph import Graph
import numpy as np
import random

# -------------Reference -------------------------------#
# Duncan J. Watts and Steven H. Strogatz,
# Collective dynamics of small-world networks,
# Nature, 393, pp. 440--442, 1998.
# -------------Reference -------------------------------#
#Queue from https://github.com/sleepokay/watts-strogatz
class Queue(object):

    def __init__(self):
        self.nextin = 0
        self.nextout = 0
        self.data = {}

    def append(self, value):
        self.data[self.nextin] = value
        self.nextin += 1

    def pop(self):
        value = self.data.pop(self.nextout)
        self.nextout += 1
        return value

    def is_empty(self):
        return self.nextout == self.nextin
    
class watts_strogatz_graph:
    
    def __init__(self, N, k, p, weight = 1):
        if k%2 != 0:
            raise Exception('k must be even :-)')
        if N < k+1:
            raise Exception(r'set n st: n >= k + 1')
        if k < np.log(N):
            raise Exception('k >> ln(N) for connected graph')
        self.k = k
        self.N = N
        self.p = p
        self.weight = weight
        self.WS = Graph(self.N)
        self.WS.add_nn_edges(self.k, self.weight)
        self.rewire_WS_model()

    def rewire_WS_model(self):
        self.rewired_edges = []
        for edge in self.WS.list_edges:
            u, v = edge
            if len(self.WS.graph[u].keys()) >= self.N - 1:
                break
            eps = random.uniform(0,1)
            if eps < self.p:
                w = random.choice([w for w in self.WS.nodes() if w is not u 
                                   and self.WS.find_edge(u, w) == None])
                self.WS.remove_edge(u, v)
                self.WS.add_edge(u, w)
                self.rewired_edges.append([(u,v),(u,w)])

    def clustering_coefficient(self):
        C = 0
        for node, node_n in self.WS.graph.items():
            max_n_edges = (len(node_n) * (len(node_n) - 1)) / 2.0
            if max_n_edges <= 0:
                continue
            n_edges = []
            # find which  neighbors of "node" are connected to every other neighbor of "node".
            for n_node in node_n:
                #go through neighbors of neighbor of "node"
                for nn_node in self.WS.get_vertices(n_node):
                    v, w = sorted([n_node, nn_node])
                    if nn_node != node and nn_node in self.WS.get_vertices(node) and (v,w) not in n_edges:
                        n_edges.append((v, w))
            C += len(n_edges) / max_n_edges
        return C / self.N
    
    def average_path_length(self):       
        average = 0.0
        for node in self.WS.nodes():
            path_lengths = {n : -1 for n in self.WS.nodes()} 
            queue = Queue()
            queue.append(node)
            path_lengths[node] = 0
            while not queue.is_empty():
                current_node = queue.pop()
                for w in self.WS.get_vertices(current_node):
                    if path_lengths[w] == -1:
                        queue.append(w)
                        path_lengths[w] = path_lengths[current_node] + 1
            average += sum(path_lengths.values()) / (len(self.WS.nodes()))
        return average / len(self.WS.nodes())
