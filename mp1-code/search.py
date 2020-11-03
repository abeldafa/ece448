# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Rahul Kunji (rahulsk2@illinois.edu) on 01/16/2019

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,greedy,astar)
import itertools
import numpy as np
import bisect
from itertools import combinations

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "dfs": dfs,
        "greedy": greedy,
        "astar": astar,
        "extra": extra,
    }.get(searchMethod)(maze)


def bfs(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    start = maze.getStart()
    goal = maze.getObjectives()[0]
    totalSteps = 0
    path = []
    PointDists = {}
    Row = maze.getDimensions()[0]
    Column = maze.getDimensions()[1]
    for i in range(Row):
        for j in range(Column):
            PointDists[(i,j)] = 100000
    PointDists[start] = 0
    Stack = []
    Stack.append(start)
    while len(Stack)!=0:
        cur = Stack.pop(0)
        totalSteps +=1
        Dist = PointDists[cur]
        if cur == goal: continue
        for neighbor in maze.getNeighbors(cur[0],cur[1]):
            if PointDists[cur]<PointDists[neighbor]-1:
                Stack.append(neighbor)
                PointDists[neighbor] = Dist + 1
    cur = goal
    while cur != start:
        path.append(cur)
        for neighbor in maze.getNeighbors(cur[0],cur[1]):
            if PointDists[cur]==PointDists[neighbor]+1:
                cur = neighbor
                break
    path.append(start)
    path.reverse()
    return path,totalSteps

def dfs(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    start = maze.getStart()
    goal = maze.getObjectives()[0]
    totalSteps = 0
    path = []
    PointDists = {}
    Row = maze.getDimensions()[0]
    Column = maze.getDimensions()[1]
    for i in range(Row):
        for j in range(Column):
            PointDists[(i,j)] = 100000
    PointDists[start] = 0
    Stack = []
    Stack.append(start)
    while len(Stack)!=0:
        cur = Stack.pop(-1)
        totalSteps +=1
        Dist = PointDists[cur]
        if cur == goal: continue
        for neighbor in maze.getNeighbors(cur[0],cur[1]):
            if PointDists[cur]<PointDists[neighbor]-1:
                Stack.append(neighbor)
                PointDists[neighbor] = Dist + 1
    cur = goal
    while cur != start:
        path.append(cur)
        for neighbor in maze.getNeighbors(cur[0],cur[1]):
            if PointDists[cur]==PointDists[neighbor]+1:
                cur = neighbor
                break
    path.append(start)
    path.reverse()
    return path,totalSteps


def greedy(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    start = maze.getStart()
    goal = maze.getObjectives()[0]
    totalSteps = 0
    path = []
    PredictDistToGoal = distDict(maze)
    PointDists = {}
    Row = maze.getDimensions()[0]
    Column = maze.getDimensions()[1]
    for i in range(Row):
        for j in range(Column):
            PointDists[(i,j)] = 100000
    PointDists[start] = 0
    Stack = []
    Stack.append(start)
    while len(Stack)!=0:
        cur = Stack.pop(0)
        totalSteps +=1
        Dist = PointDists[cur]
        if cur == goal: break
        for neighbor in GreedySort(PredictDistToGoal,maze.getNeighbors(cur[0],cur[1])):
            if PointDists[cur]<PointDists[neighbor]-1:
                Stack.append(neighbor)
                PointDists[neighbor] = Dist + 1
    cur = goal
    while cur != start:
        path.append(cur)
        for neighbor in maze.getNeighbors(cur[0],cur[1]):
            if PointDists[cur]==PointDists[neighbor]+1:
                cur = neighbor
                break 
    path.append(start)
    path.reverse()
    return path,totalSteps
    
def GreedySort(PredictDistToGoal,Neighbors):
    for i in range(len(Neighbors)-1):
        for j in range(len(Neighbors)-1):
            if PredictDistToGoal[Neighbors[j]] > PredictDistToGoal[Neighbors[j+1]]:
                temp = Neighbors[j]
                Neighbors[j] = Neighbors[j+1]
                Neighbors[j+1] = temp
    return Neighbors

def distDict(maze):
    Row = maze.getDimensions()[0]
    Column = maze.getDimensions()[1]
    PredictDistToGoal = {}
    goal = maze.getObjectives()[0]
    for i in range(Row):
        for j in range(Column):
            PredictDistToGoal[(i,j)] = CheckDist(goal,(i,j))
    return PredictDistToGoal

def CheckDist(Objective, Now):
    return abs(Objective[0]-Now[0]) + abs(Objective[1]-Now[1])


def getKey(Open, dict):
    tempList = []
    for point in Open:
        tempList.append((point,dict[point]))
    minValue = 1000000
    minKey = 0
    for kv in tempList:
        if kv[1] <minValue:
            minValue = kv[1]
            minKey = kv[0]
    return minKey

def astar(maze):
    if len(maze.getObjectives()) == 1:
        return astarOneGoal(maze)
    else:
        return astarManyGoals(maze)

def astarManyGoals(maze):
    # get distances between all nodes 
    # and save paths from nodes to nodes
    start = maze.getStart()
    goal = maze.getObjectives()
    nodes = [start] + goal
    distanceDict = np.zeros((len(nodes),len(nodes))) # store length of shortest path
    # pathDict = np.zeros(((len(nodes)),len(nodes))) # store shortest path
    pathDict = {}
    statesDict = np.zeros(((len(nodes)),len(nodes)))

    # question? Should I store the node itself?
    # can I use index in set?
    indexes = [i for i in range(len(nodes))]
    for index1, index2 in itertools.combinations(indexes,2):
        path, distance,states = getDistanceBetweenTwoNodes(nodes[index1],nodes[index2],maze)
        distanceDict[index1][index2] = distance
        distanceDict[index2][index1] = distance
        pathDict[(index1,index2)] = path
        pathDict[(index2,index1)] = reversed(path)
        statesDict[index1][index2] = states
        statesDict[index2][index1] = states
    print("211")
    # calculate the shortest paths
    objPath, totalDist = shortestPathTroughAllNodes(distanceDict)
    # print the paths
    result = []
    totalStates = 0
    print("217")
    for start,end in objPath:
        totalStates += statesDict[start][end]
        pathBetween = pathDict[(start,end)]
        for eachStep in pathBetween:
            if len(result) == 0 or eachStep != result[-1]:
                result.append(eachStep)
    print(result)
    return result,totalStates


def newDistDict(goal,maze):
    Row = maze.getDimensions()[0]
    Column = maze.getDimensions()[1]
    PredictDistToGoal = {}
    for i in range(Row):
        for j in range(Column):
            PredictDistToGoal[(i,j)] = CheckDist(goal,(i,j))
    return PredictDistToGoal

# input: start, end, maze
# output: path,shortestDistance, totalStates
# not tested yet, but done
def getDistanceBetweenTwoNodes(start, goal, maze):
    totalStates = 0
    path = []
    PredictDistToGoal = newDistDict(goal,maze)
    fn = distDict(maze)
    PointDists = {}
    Row = maze.getDimensions()[0]
    Column = maze.getDimensions()[1]
    for i in range(Row):
        for j in range(Column):
            PointDists[(i,j)] = 100000
    PointDists[start] = 0
    Open = set()
    Open.add(start)
    visited = set()
    parent = {}
    while Open:
        shortestPoint = getKey(Open,fn)
        totalStates += 1
        Open.remove(shortestPoint)
        if (shortestPoint == goal):
            break
        
        for child in maze.getNeighbors(shortestPoint[0],shortestPoint[1]):
            if PointDists[shortestPoint]<PointDists[child]-1:
                PointDists[child] = PointDists[shortestPoint]+1
            newFn = PredictDistToGoal[child] + PointDists[child]

            if child in Open:
                if newFn < fn[child]:
                    parent[child] = shortestPoint
                    fn[child] = newFn
            if child in visited:
                continue

            if (child not in visited) and (child not in Open):
                parent[child] = shortestPoint
                Open.add(child)
                fn[child] = newFn
        visited.add(shortestPoint)
    cur = goal
    while cur != start:
        path.append(cur)
        cur = parent[cur]  
    path.append(start)
    path.reverse()
    return path, fn[goal],totalStates

# find the shortest path that walks through all the objectives
# input: distance matrix, np array
# output: the order of objective, and the total distance
def shortestPathTroughAllNodes(distanceDict):
    length = len(distanceDict)
    indexes = np.arange(length)
    dp = {}
    pathDp = {}
    # 初始化基础情况
    for index in indexes:
    	tempSet = [index]
    	dp[(index,str(tempSet))] = 0
    	pathDp[(index,str(tempSet))] = [index]
    # 从小comb开始循环，再对不在其中的点循环
    # 然后套用公式
    print("before xun huan")
    totalWork = 0
    for size in range(length-1):
        for comb in combinations(indexes,size+1):
            for index in indexes:
                totalWork += len(comb)
    print("total work is ",totalWork)
    curWork = 0
    for size in range(length-1):
    	print(curWork,"/",totalWork,curWork/totalWork)
    	for comb in combinations(indexes,size+1):
    		for index in indexes:
    			if index in comb:
    				continue
    			else:
    				tempList = list(comb)
    				bisect.insort_left(tempList,index,0,len(tempList))
    				tempMin = 100000
    				tempStart = -1
    				for k in comb:
    					tempLen = dp[(k,str(list(comb)))]+distanceDict[k][index]
    					curWork+=1
    					if tempMin > tempLen:
    						tempMin = tempLen
    						tempStart = k
    				dp[(index,str(tempList))] = tempMin
    				pathDp[(index,str(tempList))] = [index]+pathDp[(tempStart,str(list(comb)))]
    print("after xun huan")
    path = pathDp[(0,str(list(indexes)))]
    result = []
    cur = 0
    while cur < len(path)-1:
        result.append((path[cur],path[cur+1]))
        cur += 1
    totalDistance = dp[(0,str(list(indexes)))]
    return result,totalDistance




def astarOneGoal(maze):
    path,states,shortestDistance = astarOneGoalPathAndDistance(maze)
    return path, states


def astarOneGoalPathAndDistance(maze):
    start = maze.getStart()
    goal = maze.getObjectives()[0]
    totalSteps = 0
    path = []
    PredictDistToGoal = distDict(maze)
    fn = distDict(maze)
    PointDists = {}
    Row = maze.getDimensions()[0]
    Column = maze.getDimensions()[1]
    for i in range(Row):
        for j in range(Column):
            PointDists[(i,j)] = 100000
    PointDists[start] = 0
    Open = set()
    Open.add(start)
    visited = set()
    parent = {}
    while Open:
        shortestPoint = getKey(Open,fn)
        totalSteps += 1
        Open.remove(shortestPoint)
        if (shortestPoint == goal):
            break
        
        for child in maze.getNeighbors(shortestPoint[0],shortestPoint[1]):
            if PointDists[shortestPoint]<PointDists[child]-1:
                PointDists[child] = PointDists[shortestPoint]+1
            newFn = PredictDistToGoal[child] + PointDists[child]

            if child in Open:
                if newFn < fn[child]:
                    parent[child] = shortestPoint
                    fn[child] = newFn
            if child in visited:
                continue

            if (child not in visited) and (child not in Open):
                parent[child] = shortestPoint
                Open.add(child)
                fn[child] = newFn
        visited.add(shortestPoint)
    cur = goal
    while cur != start:
        path.append(cur)
        cur = parent[cur]  
    path.append(start)
    path.reverse()
    return path,totalSteps, fn[goal]


def extra(maze):
    path,states,shortestDistance = extrahelper(maze)
    return path, states


def extrahelper(maze):
    start = maze.getStart()
    goal = maze.getObjectives()[0]
    totalSteps = 0
    path = []
    PredictDistToGoal = distDictextra(maze)
    fn = distDictextra(maze)
    PointDists = {}
    Row = maze.getDimensions()[0]
    Column = maze.getDimensions()[1]
    for i in range(Row):
        for j in range(Column):
            PointDists[(i,j)] = 100000
    PointDists[start] = 0
    Open = set()
    Open.add(start)
    visited = set()
    parent = {}
    while Open:
        shortestPoint = getKey(Open,fn)
        totalSteps += 1
        Open.remove(shortestPoint)
        if (shortestPoint == goal):
            break
        
        for child in maze.getNeighbors(shortestPoint[0],shortestPoint[1]):
            if PointDists[shortestPoint]<PointDists[child]-1:
                PointDists[child] = PointDists[shortestPoint]+1
            newFn = PredictDistToGoal[child] + PointDists[child]

            if child in Open:
                if newFn < fn[child]:
                    parent[child] = shortestPoint
                    fn[child] = newFn
            if child in visited:
                continue

            if (child not in visited) and (child not in Open):
                parent[child] = shortestPoint
                Open.add(child)
                fn[child] = newFn
        visited.add(shortestPoint)
    cur = goal
    while cur != start:
        path.append(cur)
        cur = parent[cur]  
    path.append(start)
    path.reverse()
    return path,totalSteps, fn[goal]


def distDictextra(maze):
    Row = maze.getDimensions()[0]
    Column = maze.getDimensions()[1]
    PredictDistToGoal = {}
    goal = maze.getObjectives()[0]
    for i in range(Row):
        for j in range(Column):
            PredictDistToGoal[(i,j)] = CheckDist(goal,(i,j))
    for i in range(Row):
        for j in range(Column):

            if i in range(int(Row/2),Row) and j in range(int(Column/2)):

                PredictDistToGoal[(i,j)] = CheckDist((int(Row/2),int(Column/2)),(i,j))
            if i in range(int(Row/2),Row) and j in range(int(Column/2),Column):

                PredictDistToGoal[(i,j)] = CheckDist((Row,Column),(i,j))
                
    return PredictDistToGoal



# def iddfs(maze):
#     start = maze.getStart()
#     goal = maze.getObjectives()[0]
#     Row = maze.getDimensions()[0]
#     Column = maze.getDimensions()[1]
    
#     getout = 0
#     for depth in range(getdist(start, goal), getdis(start, goal)):
#         if(getout == 1):
#             break
#         totalSteps = 0    
#         visited = {}
#         for i in range(Row):
#             for j in range(Column):
#                 visited[(i,j)] = 100000
#         visited[start] = 0
#         Stack = []
#         Stack.append(start)
#         while len(Stack)!=0:
#             cur = Stack.pop(-1)
#             totalSteps +=1
#             if(totalSteps >= depth):
#                 continue
#             for neighbor in maze.getNeighbors(cur[0],cur[1]):
#                 if visited[cur]<visited[neighbor]-1:
#                     Stack.append(neighbor)
#                     visited[neighbor] = visited[cur] + 1 
                    
#             if cur == goal:
#                 getout = 1
#                 continue
#     cur = goal
#     while cur != start:
#         Stack.append(cur)
#         for neighbor in maze.getNeighbors(cur[0],cur[1]):
#             if visited[cur]==visited[neighbor]+1:
#                 cur = neighbor
#                 break
#     Stack.append(start)
#     Stack.reverse()            
                                   
#     return Stack,totalSteps


# def getdis(Objective, Now):
#     return abs(Objective[0]-Now[0]) * abs(Objective[1]-Now[1])
# def getdist(Objective, Now):
#     return abs(Objective[0]-Now[0]) + abs(Objective[1]-Now[1])