from graph import Graph
from bisect import bisect_right
import random

# Inputs
# N : Number of nodes
# Delta_t: time interval
# p_Delta: link probability (local attachment)
# p_r: random link probability (global attachment)
# p_d: Node deletion probability (death rate)
# delta: weight increase
# w_0:
# seed : 

class WSN(object):
    def __init__(self, parms):
#         if len(parms) != 10:
#             raise Exception('missing inputs')
        self.N = parms.get('N')
        self.delta = parms.get('delta')
        self.Delta_t = parms.get('Delta_t')
        self.p_Delta = parms.get('p_Delta')
        self.p_r = parms.get('p_r')
        self.p_d = parms.get('p_d')
        self.w_0 = parms.get('w_0')
        self.max_time_step = parms.get('max_time_step')
        self.seed = parms.get('seed')
        self.G = Graph(self.N) 
    
    def get_node_i_prob_distrib(self, node, degree):
        return self.G.get_neighbors_weights(node) / degree
    
    def get_node_j_prob_distrib(self, node_i, node_j, degree):
        return [self.G.get_neighbors_weights(node)] / degree
    
    def select_neighbor(self, degree, zip_nn_pd):
        zip_list = [pair for pair in zip_nn_pd]
        chosen_neighb = len(zip_list)
        while chosen_neighb == len(zip_list):
            r = random.uniform(0, 1)
            temp_sort = sorted(zip_list, key = lambda x: x[1])
            node_neighb, sorted_prob_d = [[n[i] for n in temp_sort] for i in [0,1]]
            chosen_neighb = bisect_right(sorted_prob_d, r*degree)
        return node_neighb[chosen_neighb]
    
    def sort_nodes(self, node1, node2):
        n_u, n_v = sorted([node1, node2])
        return [n_u, n_v]
    
    def Attach_Nodes_and_Enhance_Edges_LA(self):
        for attachment in self.LA_todo_connect_nodes:
            node_i, node_j = attachment
            self.G.add_edge(node_i, node_j, self.w_0)
            
        for enhancement in self.LA_todo_enhancement:
            node_i, node_j = enhancement
            self.G.enhance_edge_weight(node_i, node_j, self.delta)
    
    def Attach_nodes_GA(self):
        for attachment in self.GA_todo_connect_nodes:
            node_i, node_j = attachment
            self.G.add_edge(node_i, node_j, self.w_0)
        
        
    def LocalAttachment(self):
        self.LA_todo_enhancement = []
        self.LA_todo_connect_nodes = []
        for n_i in self.G.nodes():
            i_neig = self.G.get_neighbors(n_i)
            i_neig_w = self.G.get_neighbors_weights(n_i)
            d_i = self.G.node_degree(n_i)
            if len(i_neig) > 0:
                prob_distrib = self.get_node_i_prob_distrib(n_i, d_i)
                # search first neighbor j
                n_j = self.select_neighbor(d_i, zip(i_neig, prob_distrib))
                self.LA_todo_enhancement.append(self.sort_nodes(n_i, n_j))
                j_neig = self.G.get_neighbors(n_j)
                # search second neighbor k
                if len(j_neig) >1:
                    j_neig_w = self.G.get_neighbors_weights(n_j)
                    d_j = self.G.node_degree(n_j) - self.G.graph[n_i][n_j]
                    j_neig_w_no_i = [x for x in zip(j_neig, j_neig_w) if x[0] is not n_i]
                    temp_zip = [(x[0], x[1]/d_j) for x in j_neig_w_no_i]
                    n_k = self.select_neighbor(d_j, temp_zip)
                    self.LA_todo_enhancement.append(self.sort_nodes(n_j, n_k))
                    # if node i already connected to node k, enhance connection
                    if self.G.find_edge(n_i, n_k):
                        self.LA_todo_enhancement.append(self.sort_nodes(n_i, n_k))
                    else:
                        # if missing connection, connect node i to node k with p_Delta
                        r = random.uniform(0, 1)
                        if r < self.p_Delta:
                            self.LA_todo_connect_nodes.append(self.sort_nodes(n_i, n_k))
    
    def GlobalAttachment(self):
        self.GA_todo_connect_nodes = []
        for n_i in self.G.nodes():
            r = random.uniform(0, 1)
            i_neig = self.G.get_neighbors(n_i)
            Nneigh = len(i_neig)
            if (Nneigh == 0) or (r < self.p_r and Nneigh != self.N-1):
                candidates = [n for n in self.G.nodes() if n not in i_neig]
                n_j = random.choice(candidates)
                self.GA_todo_connect_nodes.append([n_i, n_j])

    def NodeDeletion(self):
        for n_i in self.G.nodes():
            r = random.uniform(0, 1)
            if r < self.p_d:
                self.G.remove_node(n_i)
                self.G.create_new_node()
                
    def Global_and_Local_Attachment(self):
        self.GlobalAttachment()
        self.Attach_nodes_GA()
        self.LocalAttachment() 
        self.Attach_Nodes_and_Enhance_Edges_LA()
        
    def generate_WSN_graph(self):
        for time_step in range(self.max_time_step):
            self.Global_and_Local_Attachment()
            self.NodeDeletion()