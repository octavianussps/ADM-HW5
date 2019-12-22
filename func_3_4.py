
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