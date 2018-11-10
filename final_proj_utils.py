import copy

def get_leaves_to_root(edges,root_node = 1):
    
    """ Function that takes the edges of a graph and returns
        a dictionary of forward edges - i.e leaves to root"""
        
    edges_temp = copy.copy(edges)    
    leaves_to_root = {} #Edges for calculating forward messages
    
    while len(edges_temp)>2:
            node_list = list(edges_temp.keys())
            #First node is considered to be root node
            fringe = [root_node]  # this is a list of nodes queued up to be visited next
            
            visited = {node:False for node in node_list} #tracks which nodes are visited
            leaves = {node:False for node in node_list} #tracks which nodes are leaves
            leaves[root_node] = False
            
            while len(fringe) > 0:
                node = fringe.pop(0)
                visited[node] = True
                for neighbor in edges_temp[node]:
                    if not visited[neighbor] :
                        if edges_temp[neighbor] == [node]:
                            leaves[neighbor] = True
                        fringe.append(neighbor)
    
            ###########Leaf and non leaf nodes############
            leaf_nodes = [x for x in leaves.keys() if leaves[x] == True]
            #non_leaf_nodes = [x for x in leaves.keys() if leaves[x] == False]
            
            ##########Get path from leaves to nodes#########
            
            for node in leaf_nodes:
                leaves_to_root[node]=edges_temp[node][0]  
                del edges_temp[node]
            
            #Update edges list
            for node in edges_temp.keys():
                edges_temp[node] = [x for x in edges_temp[node] if x not in leaf_nodes]    
            
    ##Final edge
    non_root_node = [x for  x in edges_temp.keys() if x != root_node]
    if non_root_node != []:
        leaves_to_root[non_root_node[0]] = edges_temp[non_root_node[0]][0]
        
    return(leaves_to_root)


def get_root_to_leaves(edges):
    """ Get edges from root to leaves for backward messages"""
    edges_temp = copy.copy(edges) 
    node_list = list(edges_temp.keys())
    fringe = [node_list[0]]  # this is a list of nodes queued up to be visited next
    
    visited = {node:False for node in node_list} #tracks which nodes are visited
    visited_list= []
    root_to_leaves = {x:[] for x in node_list}
    
    while len(fringe) > 0:
        node = fringe.pop(0)
        visited[node] = True
        visited_list.append(node)
        for neighbor in edges_temp[node]:
            if not visited[neighbor] and neighbor not in visited_list:
                root_to_leaves[node].append(neighbor)
                fringe.append(neighbor)
    
    root_to_leaves = {x:y for (x,y) in root_to_leaves.items() if len(y)!= 0}
    return(root_to_leaves)
    

def get_marginals(edges,root_node):
    
    """Given edges of a tree and a root node ,return marginal distribution of the
       root node. Other variables are accessed from the global environment """
     
    leaves_to_root = get_leaves_to_root(edges,root_node)
    
    ####Get root to leaves for backward messages####
    #root_to_leaves = get_root_to_leaves(edges)
    reviewed_edges = []
    
    #####Calculate forward messages i.e. leaves to root #####
    
    for item in leaves_to_root.items():
         messages[item] = node_potential_array[item[0]] 
         #Get end points of reviewed edges
         #end_points = [y for (x,y) in reviewed_edges]
         ##Get messages from neighboring edges
         neighboring_edges = [(x,y)  for (x,y) in reviewed_edges if y == item[0]]
         ##product of messages from neighbors
         message_from_neighbors = np.prod([message for (edge,message) in 
                                           messages.items() if edge in neighboring_edges ],axis=0)
    
         messages[item] = np.prod([messages[item] ,message_from_neighbors],axis=0)@ edge_potential_matrix[item]

         reviewed_edges.append(item)
    
    marginal = np.prod([y for (x,y) in messages.items() if x[1] == root_node],axis=0)
    marginals[root_node] = marginal/np.sum(marginal)