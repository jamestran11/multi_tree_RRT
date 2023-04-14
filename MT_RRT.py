"""
tRoot will be from xstart always
heuristic trees will be generated randomly from the RANDOMGENERATE function and put into some datastructure

when the tRoot is close to another heuristic tree, we make a connection

1) CONNECT-NODES
we sample random nodes in the free space.
    FIrst check if the rand node is within the distance threshold of the tRoot
    if it is, we add it to the tRoot tree via EXTEND function
    if not, check if it is near any of the heuristic trees.
    if it is, add it to the heuristic tree via ADD_NODE

    if no tree is close, a new heuristic tree will be created from the random node vis RANDOMGENERATE
    this tree will be added to the set of heuristic trees via ADDTREE

2) CONNECT-TREES
Check if two trees are close 
we need to get the two trees and teh two adjacent nodes through C_Trees somehow
if two trees are close enough, merge them
if one of the trees is actually tRoot, the other tree will act as the heuristic
    this will bias/guide the next random node through HEURISTIC_STATE
    (probably create X new somewhere near)
    tRoot will be extended with x new via EXTEND function
    the heuristic tree will be removed**

"""
import networkx as nx
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from PIL import Image



def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def distanceFromNodeToTree(node, tree):
    closestNode = None
    shortestDistance = 10000000
    listOfNodes = list(tree.nodes)
    for treeNode in listOfNodes:
        if euclidean_distance(node, treeNode) < shortestDistance:
            shortestDistance = euclidean_distance(node, treeNode)
            closestNode = treeNode
    return (closestNode, shortestDistance)

def getNearestNeighbor(t_tree, x_rand, threshold):
    nearestNeighbor = None
    listOfNodes = list(t_tree.nodes)
    shortestDistance = 100000000
    for node in listOfNodes:
        distance = euclidean_distance(node, x_rand)
        if distance < shortestDistance:
            shortestDistance = distance
        if shortestDistance < threshold:
            nearestNeighbor = node

    return nearestNeighbor

def getXNew(node, x_rand, max_distance):
    # Calculate the distance between the two points
    distance = math.sqrt((x_rand[0] - node[0])**2 + (x_rand[1] - node[1])**2)
    
    # Calculate the fraction of the distance that is within the max distance
    if distance <= max_distance:
        fraction = 1
    else:
        fraction = max_distance / distance
    
    # Calculate the x and y coordinates of the midpoint
    x = node[0] + fraction * (x_rand[0] - node[0])
    y = node[1] + fraction * (x_rand[1] - node[1])
    
    # Return the midpoint
    return (round(x), round(y))

#collision is if cell has value 0
def isPathCollisionFree(node1, node2, map):
    x0 = node1[0]
    y0 = node1[1]
    x1 = node2[0]
    y1 = node2[1]

    coords = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while x0 != x1 or y0 != y1:
        coords.append((x0, y0))
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    coords.append((x0, y0))

    for coord in coords:
        if map[coord[0]][coord[1]] == 0:
            return False
    return True


def Extend(t_tree, x_rand, map):
    x_near = getNearestNeighbor(t_tree, x_rand,2)
    x_new = getXNew(x_near,x_rand, 2)
    if(isPathCollisionFree(x_near, x_new, map)):
        t_tree.add_node(x_new)
        t_tree.add_edge(x_near, x_new)
        return x_new
    return None

def getFreeSpace(arr):
    indices = np.argwhere(arr == 1)
    return [(i, j) for i, j in indices]

def RandomGenerate(map):
    freeCells = getFreeSpace(map)
    random_index = random.randint(0, len(freeCells) - 1)
    treeRoot = nx.Graph()
    treeRoot.add_node(freeCells[random_index])
    return treeRoot

def RandomState(map):
    freeCells = getFreeSpace(map)
    random_index = random.randint(0, len(freeCells) - 1)
    node = freeCells[random_index]
    return node

#Returns the (nxGraph from start, the branch node from that start graph 
# , the nxGraph closest to that start graph, and the node closest to the start branch node)
def twoTreesAreClose(t_root, listOfHeuristicTrees):
    for ogTreeNode in list(t_root.nodes):
        for nxGraph in listOfHeuristicTrees:
            allNodes = list(nxGraph.nodes)
            for node in allNodes:
                if euclidean_distance(ogTreeNode, node) < 2:
                    return (t_root, ogTreeNode, nxGraph, node)

    return None


def HeuristicState(tree, map):
    #create bounding box of tree, bottom right node and top right node
    #randomly get a free space cell in the bounding box
    listOfNodes = list(tree.nodes)
    listOfNodes = np.array(listOfNodes)

    # Get the indices of the leftmost, rightmost, topmost, and bottommost points
    left_idx = np.argmin(listOfNodes[:, 0])
    right_idx = np.argmax(listOfNodes[:, 0])
    top_idx = np.argmin(listOfNodes[:, 1])
    bottom_idx = np.argmax(listOfNodes[:, 1])

    # Get the leftmost, rightmost, topmost, and bottommost points
    leftmost = tuple(listOfNodes[left_idx])
    rightmost = tuple(listOfNodes[right_idx])
    topmost = tuple(listOfNodes[top_idx])
    bottommost = tuple(listOfNodes[bottom_idx])

    subMap = map[topmost[1]:bottommost[1], leftmost[0]:rightmost[0]]
    x_rand = RandomState(subMap)

    return x_rand

def isCloseToGoal(node, goalNode, dist):
    if euclidean_distance(node, goalNode) < dist:
        return True
    return False

def ExtendTree(tree1, tree2,node1,node2):
    combinedTree = nx.union(tree1,tree2)
    combinedTree.add_edge(node1,node2)
    return combinedTree

def MT_RRT(x_start, x_goal, map, dist):
    N = 10
    n = 0
    ogTree = nx.Graph()
    ogTree.add_node(x_start)
    #right now only one, should create a list of them
    listOfHeuristicTrees = []
    firstHeuristicTree = RandomGenerate(map)
    listOfHeuristicTrees.append(firstHeuristicTree)
    while n <= N:
        n += 1
        print(n)
        if twoTreesAreClose(ogTree, listOfHeuristicTrees) == None:
            addedNewInfo = False
            x_rand = RandomState(map)
            #compare x_rand to root of tree? or check all nodes of tree and find closest
            if distanceFromNodeToTree(x_rand, ogTree)[1] < dist:
                x_new = Extend(ogTree, x_rand, map)
                addedNewInfo = True

            if addedNewInfo == False:
                for tree in listOfHeuristicTrees:
                    closestNode, distanceFromNode = distanceFromNodeToTree(x_rand, tree)
                    if distanceFromNode < dist:
                        if isPathCollisionFree(x_rand, closestNode, map):
                            tree.add_node(x_rand)
                            tree.add_edge(closestNode,x_rand)
                            addedNewInfo == True
            if addedNewInfo == False:
                newHeuristicTree = RandomGenerate(map)
                newHeuristicTree.add_node(x_rand)
                listOfHeuristicTrees.append(newHeuristicTree)

        if twoTreesAreClose(ogTree, listOfHeuristicTrees) != None:
            #should loop through all listOfHeuristicTrees?
            Ttree1, node1, Ttree2, node2 = twoTreesAreClose(ogTree, listOfHeuristicTrees)
            if(Ttree1 == ogTree):
                x_rand = HeuristicState(Ttree2, map)
                x_new = Extend(ogTree, x_rand, map)
                listOfHeuristicTrees.remove(Ttree2)
                if isCloseToGoal(x_new, x_goal):
                    ogTree.add_edge(x_new, x_goal)
                    return ogTree
            else:
                if isPathCollisionFree(node1, node2, map):
                    combinedTree = ExtendTree(Ttree1, Ttree2)
                    listOfHeuristicTrees.append(combinedTree)
                    listOfHeuristicTrees.remove(Ttree2)
    return



occupancy_map_img = Image.open('./map.png')
occupancy_grid_raw = (np.asarray(occupancy_map_img) > 0).astype(int)
#print(euclidean_distance((0,0),(1,1)))
path = MT_RRT((184,170),(184,207),occupancy_grid_raw, 2)
print(path)