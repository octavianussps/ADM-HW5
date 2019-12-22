'''Functionality 1 '''
#!/usr/bin/env python
# coding: utf-8
# # Functionality 1 - Find the Neighbours!
# Take input:
# a node v
# One of the following distances function: 
# t(x,y), d(x,y) or network distance .
# a distance threshold d.

# Implement an algorithm (using proper data structures) that returns :

# the set of nodes at distance <= d from v, corresponding to v’s neighborhood.

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


df_D = pd.read_csv('./DATA/Distance_graph',header=0, sep=',',
                index_col=None, encoding = "ISO-8859-1")
df_D.columns=["Distance"]
df_D.drop([0,1,2,3,4,5],inplace=True)
df_D.reset_index(drop=True)
df_D.head(10)



df_D=df_D['Distance'].str.split(" " , expand=True)
df_D.columns=['a','Node1','Node2','Distance']
df_D['Node1']=pd.to_numeric(df_D['Node1'])
df_D['Node2']=pd.to_numeric(df_D['Node2'])
df_D['Distance']=pd.to_numeric(df_D['Distance'])
df_D=df_D.reset_index(drop=True)
df_D.head(4)

df_T = pd.read_csv('./DATA/Travel_time_graph',header=0, sep=',',
                index_col=None, encoding = "ISO-8859-1")
df_T.columns=["Distance"]
df_T.drop([0,1,2,3,4,5],inplace=True)
df_T.head(5)


# In[6]:


df_T=df_T["Distance"].str.split(" " , expand=True)
df_T.columns=['a','Node1','Node2','Distance']
df_T['Node1']=pd.to_numeric(df_T['Node1'])
df_T['Node2']=pd.to_numeric(df_T['Node2'])
df_T['Distance']=pd.to_numeric(df_T['Distance'])
df_T=df_T.reset_index(drop=True)
df_T.head(5)


# In[7]:


df_T.shape


# # Search Neighbours:
# 

# In[49]:


node= int(input("Enter node numbner "))
distance_type=input("Enter one of this Distance type: Time or Physical ")
threshold=int(input("Enter your threshold "))  
#The idea is to iterate all direct neighbour from node v
#and check the values Corresponding to the input
#and we recursively subtract the new distance from initial threshold 
#to check the neighbours of the initial node's neighbour's Satisfying the threshold
def search_Neighbours(node, distance_type, threshold): 
    if distance_type == "Physical":
        data = df_D
    elif distance_type == "Time":
        data=df_T   
    out_df = data[(data['Node1'] == node) & (data['Distance'] <= threshold)]

    Neighbours=list(out_df.Node2.values)
        #print(Neighbours)
    for j in Neighbours:
        neigh_dis = data[(data['Node1'] == node) & (data['Node2'] == j)]
        neigh_dis=int(neigh_dis['Distance'])
        out_df=pd.concat([out_df,search_Neighbours(j, distance_type, threshold-neigh_dis)])
    return out_df

out_df=search_Neighbours(node,distance_type,threshold)
out_df.head(7)


all_Neighbours = list(set(out_df['Node2'].values))
all_Neighbours


import folium
from folium import plugins
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


df_c = pd.read_csv('./DATA/Coordinates',header=0, sep=',',
                index_col=None, encoding = "ISO-8859-1")
df_c.columns=["Coordinates"]
df_c.drop([0,1,2,3,4,5],inplace=True)
df_c.head(3)


# In[54]:


df_c=df_c.Coordinates.str.split(" " , expand=True)
df_c.columns=['v','Node_ID','long','lat']
df_c['Node_ID']=pd.to_numeric(df_c['Node_ID'])
df_c['long']=pd.to_numeric(df_c['long'])
df_c['lat']=pd.to_numeric(df_c['lat'])
df_c=df_c.reset_index(drop=True)
df_c.head(5)


df_c['lat']=df_c['lat']/1000000
df_c['long']=df_c['long']/1000000




# plotting the starting node and node within the threshold
for i in range(len(out_df_cor)):
    folium.CircleMarker(location=[lat[i],long[i]],
                        radius=5,
                        popup=['Node_ID'],
                        fill_color="#3db7e4",
                       ).add_to(m)
                       

''' Functionality 2'''
import pandas as pd
import networkx as nx
from collections import defaultdict
from itertools import permutations

# Function to get the data
def GetData (path):
    data = pd.read_csv(path, header=0, sep=',', 
                              index_col=None, encoding = "ISO-8859-1")
    data.columns = ["column"]
    data = data.iloc[6:]
    data = data.reset_index(drop = True)
    result = data.column.str.split(" ", expand = True)
    return result

# Function to get the shortest path between 2 nodes
def ShortestPath(start, end, g):
    #init. put start as next list
    path = []
    next_ = [start] 
    passed = []
    i = 0
    while end not in next_:
        if next_[i] not in passed:
            for j in list(g.neighbours(next_[i])):
                next_.append[j]
            path.append(list(g.neighbours(next_[i])))
            passed.append(next_[i])
        i += 1
    added = passed[-1]
    i = 1
    result = [end, added]
    while i < len(g) and added != start:
        parent = []
        for i in range(len(path)):
            for j in range(len(path[i])):
                if path[i][j] == added:
                    parent.append(i)
        added = passed [parent[0]]
        i += 1
        result.append(added)
    result.reverse()
    return (result)

# Function to calculate distance between nodes in edges
def NodeDiff (start, end, func):
    result = func.loc[(dist[1] == start) & (func[2] == end)]
    if len(result) > 0: return int(result[3])
    else: return 0

# Function to calculate distance from start to end
def TotalDist (start, end, func):
    path = ShortestPath(start, end ,g)
    result = 0
    i = 0
    while i < len(path) - 1:
        result = result + NodeDiff(path[i], path[i+1], func)
        i += 1
    return (result)

def SmartestNetwork (inputs, func):
    lists = []
    for item in permutations(inputs):
        poss = list(item)
        tot = 0
        i = 0
        while i < len(poss)-1:
            tot = tot + TotalDist(poss[i], poss[i+1], func)
            i+=1
        possMaps = (tot, poss)
        lists.append(possMaps)
    lists = sorted(lists, key = lambda x: (x[0]), reverse = False)
    bestPoss = lists[0][0]
    
    i = 0
    edges_ = []
    while i < len(bestPoss) - 1:
        edges_ = edges_ + ShortestPath (bestPoss[i], bestPoss[i+1], g)
        i+=1
    return([(bestPoss, edges_)])
	
coordinates = "./data/USA-road-d.CAL.co"
times = "./data/USA-road-t.CAL.gr"
distances = "./data/USA-road-d.CAL.gr"
cD = GetData(coordinates)
tD = GetData(times)
dD = GetData(distances)

g = nx.Graph()
for key, value in dD.iterrows():
    g.add_edge(value[1], value[2], dist = value[3])
    
'''Functionality 3 & 4'''
#Define a class that will help us build the graph
class Graph():
    def __init__(self):
        # dictionary with key a node and values all of its neighbors
        self.edges = {}
        # dictionary with weights between two nodes as values and the tuple of the two nodes as key
        self.weights = {}

    def add_edge(self, from_node, to_node, weight):
        if (from_node not in self.edges):
            self.edges[from_node] = [to_node]
        else:
            self.edges[from_node].append(to_node)

        if (to_node not in self.edges):
            self.edges[to_node] = [from_node]
        else:
            self.edges[to_node].append(from_node)

        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight

# Define a function for the setup. It will return the two elements we need, the sequence of nodes we want to visit and the graph.

def setup():
    # Store the needed parameters from the input
    print(
        "Write a node H, the sequence of nodes to visit in order p_1,...,p_n, and the mode you want t(time) / d(distance) / nd(network distance). Ex. '25 50,22,13,4 t'")
    H, s, mode = input().split()
    sequence = [H] + s.split(",")
    sequence = [int(x) for x in sequence]

    # Select the path for the correct opening of the data
    # We don't care what file we open if we use nd since the weight of all nodes will be 1
    if mode in ["t", "nd"]:
        mode = "USA-road-t.CAL.gr"
    elif mode == "d":
        mode = "USA-road-d.CAL.gr"
    else:
        print("No correct mode selected.")

    # Open the file and store the graph in a list
    edges = []
    with open(r"C:\Users\39335\Desktop\University\Aris\\" + mode) as file:
        for line in file:
            if line.startswith("a"):
                edges.append([int(x) for x in line[2:].split()])

    # Make the graph using the class
    graph = Graph()
    for edge in edges:
        graph.add_edge(*edge)

    return (sequence, graph, mode)

# Define the function that finds the shortes path between two nodes. It takes as input the graph, the initial point and the end point.
# If we want to use the network distance we need to pass "nd = True", otherwise the weight of the graph will be used.
# If we want to return the weight of the path set "w = True"
def walk(graph, initial, end, nd=False, w = False):
    # Shortest paths is a dictionary with keys the nodes and values a tuple of (previous node, weight)
    # We need to initialize the dictionary
    shortest_paths = {initial: (None, 0)}
    current = initial
    visited = set()  # flag the nodes already visited

    while current != end:
        visited.add(current)
        destinations = graph.edges[current]
        weight_to_current = shortest_paths[current][1]

        # For each node we visit we find the closest neighbor and update the total distance
        for next in destinations:
            # Make the distiction if we want the network distance (each weight = 1)
            if nd == False:
                weight = graph.weights[(current, next)] + weight_to_current
            elif nd == True:
                weight = 1 + weight_to_current

            if next not in shortest_paths:
                shortest_paths[next] = (current, weight)
            else:
                current_weight = shortest_paths[next][1]
                if current_weight > weight:
                    shortest_paths[next] = (current, weight)

        # We need not to take into account the visited nodes
        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Not possible"
        # The next node to visit is the destination with the lowest weight
        current = min(next_destinations, key=lambda k: next_destinations[k][1])

    # Work back through destinations in shortest path
    path = []
    while current is not None:
        path.append(current)
        next = shortest_paths[current][0]
        current = next
    # Reverse path
    path = path[::-1]

    if w == True:
        return (weight)
    return path
