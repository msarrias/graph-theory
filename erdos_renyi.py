import numpy as np
import random
from itertools import combinations 
from graph import *

class erdos_renyi_graph:
    def __init__(self, N, p, weight = 1, seed = False):
        if seed: random.seed(seed)       
        self.N = N
        self.p = p
        self.weight = weight
        self.ER = Graph(self.N, self.weight)
        self.ER.init_nodes()
        self.edges_list = []
        self.eval_list = []
        self.generate_ER_graph()
  
    def generate_ER_graph(self):
        edges = list(combinations(range(self.N), 2))
        for edge in edges:
            u,v = edge
            eps = random.uniform(0,1)
            if (eps < self.p): 
                self.ER.add_edge(u, v)