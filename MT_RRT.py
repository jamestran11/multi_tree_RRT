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
import copy



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


def Extend(t_tree, x_rand, map, closeMetric):
    distanceThreshold = closeMetric*2
    x_near = getNearestNeighbor(t_tree, x_rand, distanceThreshold)
    if x_near != None:
        distanceToExtend = closeMetric
        x_new = getXNew(x_near,x_rand, distanceToExtend)
        if(isPathCollisionFree(x_near, x_new, map)):
            t_tree.add_node(x_new, pos=x_new)
            t_tree.add_edge(x_near, x_new)
            return x_new
        else:
            print('missed opp to grow og tree, collision')
    else:
        print('missed opp to grow og tree, no nearest neighbor under threshold')
    return None

def getFreeSpace(arr):
    indices = np.argwhere(arr == 1)
    return [(i, j) for i, j in indices]

def RandomGenerate(map):
    freeCells = getFreeSpace(map)
    random_index = random.randint(0, len(freeCells) - 1)
    treeRoot = nx.Graph()
    treeRoot.add_node(freeCells[random_index], pos = freeCells[random_index])
    return treeRoot

def RandomState(map):
    #bug here: freeCells sometimes empty
    freeCells = getFreeSpace(map)
    random_index = random.randint(0, len(freeCells) - 1)
    node = freeCells[random_index]
    return node

#Returns the (nxGraph from start, the branch node from that start graph 
# , the nxGraph closest to that start graph, and the node closest to the start branch node)
def twoTreesAreClose(ogTree, listOfHeuristicTrees, closeMetric):
    allTrees = copy.copy(listOfHeuristicTrees)
    allTrees.insert(0, ogTree)
    # print("printing all trees")
    # for tree in allTrees:
    #     print(tree.nodes)
    # print("--------")
    for i in range(len(allTrees)):
        for j in range(i+1, len(allTrees)):
            tree1 = allTrees[i]
            tree2 = allTrees[j]
            for nodeFromFirstTree in list(tree1.nodes):
                for nodeFromSecondTree in list(tree2.nodes):
                    distanceBetweenTreeNodes = euclidean_distance(nodeFromFirstTree, nodeFromSecondTree)
                    if distanceBetweenTreeNodes < closeMetric:
                        return (tree1, nodeFromFirstTree, tree2, nodeFromSecondTree)

    return None


def HeuristicState(tree, map):
    #create bounding box of tree, bottom right node and top right node
    #randomly get a free space cell in the bounding box
    listOfNodes = list(tree.nodes)
    listOfNodes = np.array(listOfNodes)
    if len(listOfNodes) > 1:
        # Get the indices of the leftmost, rightmost, topmost, and bottommost points
        left_idx = np.argmin(listOfNodes[:, 1])
        right_idx = np.argmax(listOfNodes[:, 1])
        top_idx = np.argmin(listOfNodes[:, 0])
        bottom_idx = np.argmax(listOfNodes[:, 0])

        # Get the leftmost, rightmost, topmost, and bottommost points
        leftmost = tuple(listOfNodes[left_idx])
        rightmost = tuple(listOfNodes[right_idx])
        topmost = tuple(listOfNodes[top_idx])
        bottommost = tuple(listOfNodes[bottom_idx])

        subMap = map[topmost[0]:bottommost[0], leftmost[1]:rightmost[1]]
        
        #temp fix
        if(np.all(subMap==0)):
            return listOfNodes[0]
        # TODO: fix bug where sometimes there are no freecells in submap
        x_rand = RandomState(subMap)
        # TODO: Check for bugs, make sure pixels line up between submap and real map
        x_rand = (x_rand[0]+topmost[0], x_rand[1]+leftmost[1])
        return x_rand
    else:
        x_rand = listOfNodes[0]

    return x_rand

def isCloseToGoal(node, goalNode, dist):
    if euclidean_distance(node, goalNode) < dist:
        return True
    return False

def ExtendTree(tree1, tree2, node1, node2):
    combinedTree = nx.union(tree1,tree2)
    combinedTree.add_edge(node1,node2)
    return combinedTree



def MT_RRT(x_start, x_goal, map, closeMetric):
    N = 250
    n = 0
    ogTree = nx.Graph()
    ogTree.add_node(x_start,pos = x_start)
    #right now only one, should create a list of them
    listOfHeuristicTrees = []
    firstHeuristicTree = RandomGenerate(map)
    listOfHeuristicTrees.append(firstHeuristicTree)
    while n <= N:
        print("===========================")
        n += 1
        print("Iteration:" + str(n))
        # if ogTree and no other trees are close
        if twoTreesAreClose(ogTree, listOfHeuristicTrees, closeMetric) == None:
            addedNewInfo = False
            x_rand = RandomState(map)
            #check if x_rand is close to any node in ogTree, if so, extend og tree
            if distanceFromNodeToTree(x_rand, ogTree)[1] < closeMetric:
                x_new = Extend(ogTree, x_rand, map, closeMetric)
                addedNewInfo = True
                distanceConsideredCloseToGoal = 10
                if x_new != None:
                    print("x_rand is close to OG Tree, extending OG Tree")
                    print("there are currently " +str(len(ogTree.nodes))+ " nodes in the OG Tree.")
                    if isCloseToGoal(x_new, x_goal, distanceConsideredCloseToGoal):
                        ogTree.add_edge(x_new, x_goal)
                        print(x_new)
                        print(x_goal)
                        print("WINNER WINNER")
                        return ogTree, listOfHeuristicTrees

            #if we didnt extend ogTree, loop through all heuristicTrees and check if close to any
            if addedNewInfo == False:
                for tree in listOfHeuristicTrees:
                    closestNode, distanceFromNode = distanceFromNodeToTree(x_rand, tree)
                    if distanceFromNode < closeMetric:
                        if isPathCollisionFree(x_rand, closestNode, map):
                            print("x_rand node was not close to OG Tree, but was close to another tree. Adding it to this heursitic tree")
                            tree.add_node(x_rand,pos =x_rand)
                            tree.add_edge(closestNode,x_rand)
                            addedNewInfo = True
                            break

            if addedNewInfo == False:
                print("x_rand node was not close to any tree, discarding x_rand and creating a new heuristic tree somewhere")
                newHeuristicTree = RandomGenerate(map)
                listOfHeuristicTrees.append(newHeuristicTree)

        # loop though all trees and not just og tree
        if twoTreesAreClose(ogTree, listOfHeuristicTrees, closeMetric) != None:
            #should loop through all listOfHeuristicTrees?
            print("Two trees are close to each other")
            Ttree1, node1, Ttree2, node2 = twoTreesAreClose(ogTree, listOfHeuristicTrees, closeMetric)
            if(Ttree1 == ogTree):
                print("OG Tree is close to one of the heuristic trees")
                x_rand = HeuristicState(Ttree2, map)
                x_new = Extend(ogTree, x_rand, map, closeMetric)
                if x_new != None:
                    print("Removing the heuristic tree and extending og tree")
                    print("there are currently " +str(len(ogTree.nodes))+ " nodes in the OG Tree.")
                    listOfHeuristicTrees.remove(Ttree2)
                    if isCloseToGoal(x_new, x_goal, closeMetric):
                        ogTree.add_edge(x_new, x_goal)
                        print(x_new)
                        print(x_goal)
                        print("WINNER WINNER")
                        return ogTree, listOfHeuristicTrees
                else:
                    Ttree2.remove_node(node2)
                    if len(list(Ttree2.nodes)) == 0:
                        listOfHeuristicTrees.remove(Ttree2)

            else:
                print("attempting to combine trees")
                if isPathCollisionFree(node1, node2, map):
                    print("success")
                    combinedTree = ExtendTree(Ttree1, Ttree2, node1, node2)
                    listOfHeuristicTrees.append(combinedTree)
                    listOfHeuristicTrees.remove(Ttree1)
                    listOfHeuristicTrees.remove(Ttree2)
                else:
                    #Bug here: these two trees will always be close, but there is a obstacle blocking their connect
                    #Next iteration will not add any new point because these two trees still return true that they are close
                    #Temp fix is to remove the two nodes from the two trees
                    Ttree1.remove_node(node1)
                    Ttree2.remove_node(node2)
                    if len(list(Ttree1.nodes)) == 0:
                        listOfHeuristicTrees.remove(Ttree1)
                    if len(list(Ttree2.nodes)) == 0:
                        listOfHeuristicTrees.remove(Ttree2)

                    print("fail, there was a collision")

        print("===========================")

    return ogTree, listOfHeuristicTrees


im = plt.imread(r'./occupancy_map.png')
implot = plt.imshow(im)

occupancy_map_img = Image.open('./occupancy_map.png')
occupancy_grid_raw = (np.asarray(occupancy_map_img) > 0).astype(int)
start = (500,200)
goal = (375,315)
#(y,x)
# start = (184,160)
# goal = (184,210)
tree, heuristicTrees = MT_RRT(start,goal,occupancy_grid_raw, 100)

#Plot og tree nodes
print("there are " + str(len(tree.nodes)) + " nodes in the og tree")
print("there are " + str(len(heuristicTrees)) + " heuristic trees.")
yCoordinates = list(zip(*list(tree.nodes)))[0]
xCoordinates = list(zip(*list(tree.nodes)))[1]
plt.scatter(x=xCoordinates, y=yCoordinates, c='b', s=4)

#Plot start and goal positions
plt.scatter(start[1], start[0], c='g', s=4)
plt.scatter(goal[1], goal[0], c='r', s=4)

for i in range(len(heuristicTrees)):
    heuristicTree = heuristicTrees[i]
    yCoordinates = list(zip(*list(heuristicTree.nodes)))[0]
    xCoordinates = list(zip(*list(heuristicTree.nodes)))[1]
    plt.scatter(x=xCoordinates, y=yCoordinates, s=4)


plt.show()
