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
