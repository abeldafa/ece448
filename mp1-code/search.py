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
def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "dfs": dfs,
        "greedy": greedy,
        "astar": astar,
    }.get(searchMethod)(maze)


def bfs(maze):
    start = maze.getStart()
    goal = maze.getObjectives()[0]
    visited = set()
    totalSteps = 0
    path = []
    pathDict = {}

    q = []
    q.append(start)
    visited.add(start)
    while True:
        cur = q.pop(0)
        totalSteps += 1
        visited.add(cur)
        if cur==goal:
            break
        for neighbor in maze.getNeighbors(cur[0],cur[1]):
            if neighbor in visited:
                continue
            else:
                visited.add(neighbor)
                q.append(neighbor)
                pathDict[neighbor] = cur
    
    temp = goal
    while temp!=start:
        path.append(temp)
        temp = pathDict[temp]
    path.append(start)
    path.reverse()
    return path,totalSteps    


def dfs(maze):

    return [],0

def greedy(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    return [], 0


def astar(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    return [], 0
