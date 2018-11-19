from collections import defaultdict 
import networkx as nx
from networkx import DiGraph, simple_cycles
from networkx import transitivity, average_clustering
# represents a directed graph using adjacency list representation 
class Graph: 
    def __init__(self,vertices): 
        self.V= vertices # number of vertices 
        self.graph = defaultdict(list) # default dictionary to store graph 
   
    # function to add an edge to graph 
    def addEdge(self,u,v): 
        self.graph[u].append(v) 
   
    # A function used by DFS 
    def DFSUtil(self,v,visited, cur_scc): 
        # Mark the current node as visited and print it 
        visited[v]= True
        # print v,
        cur_scc.append(v)
        #Recur for all the vertices adjacent to this vertex 
        for i in self.graph[v]: 
            if visited[i]==False: 
                self.DFSUtil(i,visited, cur_scc) 
  
    def fillOrder(self,v,visited, stack): 
        # Mark the current node as visited  
        visited[v]= True
        #Recur for all the vertices adjacent to this vertex 
        for i in self.graph[v]: 
            if visited[i]==False: 
                self.fillOrder(i, visited, stack) 
        stack = stack.append(v) 
      
    # Function that returns reverse (or transpose) of this graph 
    def getTranspose(self): 
        g = Graph(self.V) 
        # Recur for all the vertices adjacent to this vertex 
        for i in self.graph: 
            for j in self.graph[i]: 
                g.addEdge(j,i) 
        return g 
   
    
    def find_SCC(self): # find all strongly connected components,
    # return number of scc, scc list, and max scc len
        # print('--------all scc--------')
        num_scc = 0
        stack = [] 
        all_scc = []
        max_len = 0
        # Mark all the vertices as not visited (For first DFS) 
        visited =[False]*(self.V) 
        # Fill vertices in stack according to their finishing 
        # times 
        for i in range(self.V): 
            if visited[i]==False: 
                self.fillOrder(i, visited, stack) 
        # Create a reversed graph 
        gr = self.getTranspose() 
        # Mark all the vertices as not visited (For second DFS) 
        visited =[False]*(self.V) 
        # Now process all vertices in order defined by Stack 
        while stack: 
            i = stack.pop() 
            if visited[i]==False: 
                cur_scc = []
                gr.DFSUtil(i, visited, cur_scc) 
                # print","
                num_scc += 1
                all_scc.append(cur_scc)
        # print('------end all scc-------')
        for i in all_scc:
            if len(i) == 1: # remove size 1 scc
                all_scc.remove(i)
                num_scc -= 1
        for i in all_scc: # clean size 1 scc again
            if len(i) == 1: 
                all_scc.remove(i)
                num_scc -= 1
        for i in all_scc: # clean size 1 scc again
            if len(i) == 1: 
                all_scc.remove(i)
                num_scc -= 1
        for i in all_scc: # compare max len
            if max_len < len(i):
                max_len = len(i)
        return num_scc, all_scc, max_len

    def all_cycles(self): # find all cycles 
    # return number of cycles, cycle list, and max cycle len
        DG = DiGraph(self.graph)
        cycle_list = list(simple_cycles(DG))
        max_len = 0
        num_cycles = len(cycle_list)
        for i in cycle_list:
            if len(i) == 1: # remove size 1 cycle
                cycle_list.remove(i)
                num_cycles -= 1
        for i in cycle_list: # clean size 1 cycle again
            if len(i) == 1:
                cycle_list.remove(i)
                num_cycles -= 1
        for i in cycle_list: # clean size 1 cycle again
            if len(i) == 1:
                cycle_list.remove(i)
                num_cycles -= 1
        for i in cycle_list: # compare max len
            if max_len < len(i): 
                max_len = len(i)
        return num_cycles, cycle_list, max_len

    def cycle_in_cycle(self): # find cycles inside cycle
        _, cycle_list, _ = self.all_cycles()
        num_incycle = 0
        for i in range(len(cycle_list)):
            in_c = set(cycle_list[i])
            for j in range(len(cycle_list)):
                out_c = set(cycle_list[j])
                if in_c < out_c: # is subset
                    num_incycle += 1
        return num_incycle

    def DFS(self,v,seen=None,path=None): # depth first search
    # return all paths from node v
        if seen is None: seen = []
        if path is None: path = [v]
        seen.append(v)
        paths = []
        for t in self.graph[v]:
            if t not in seen:
                t_path = path + [t]
                paths.append(tuple(t_path))
                paths.extend(self.DFS(t, seen[:], t_path))
        paths = [list(p) for p in paths] # convert list of tuples to list of lists
        return paths

    def longest_path(self, v): # find longest path
    # return longtest path len, and all longest paths
        paths = self.DFS(v)
        max_len = max(len(p) for p in paths)
        max_paths = [p for p in paths if len(p) == max_len]
        return max_len, max_paths

    def global_cluster_coef(self): # global clustering coefficient
        DG = DiGraph(self.graph)
        G = DG.to_undirected() # to undirected
        # print(G.edges())
        return transitivity(G)

    def av_local_cluster_coef(self):
        DG = DiGraph(self.graph)
        return average_clustering(DG)









