# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    dataStructure = util.Stack()  # Initialize a stack to keep track of nodes
    visited = set()  # Keep track of visited nodes

    # Push initial state into stack along with an empty path and cost
    dataStructure.push((problem.getStartState(), []))

    while not dataStructure.isEmpty():
        # Pop a node from the stack
        node, path = dataStructure.pop()

        # Check if the node is the goal state
        if problem.isGoalState(node):
            return path  # Return the path if goal is reached

        # If the node is not visited
        if node not in visited:
            visited.add(node)  # Mark the node as visited

            # Expand the current node and push its children into the stack
            for successor, action, cost in problem.getSuccessors(node):
                new_path = path + [action]  # Append the current action to the path
                dataStructure.push((successor, new_path))  # Push the successor and its path to the stack

    return []  # Return an empty list if no path is found

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    dataStructure = util.Queue()  # Initialize a stack to keep track of nodes
    visited = set()  # Keep track of visited nodes

    # Push initial state into stack along with an empty path and cost
    dataStructure.push((problem.getStartState(), []))

    while not dataStructure.isEmpty():
        # Pop a node from the stack
        node, path = dataStructure.pop()

        # Check if the node is the goal state
        if problem.isGoalState(node):
            return path  # Return the path if goal is reached

        # If the node is not visited
        if node not in visited:
            visited.add(node)  # Mark the node as visited

            # Expand the current node and push its children into the stack
            for successor, action, cost in problem.getSuccessors(node):
                new_path = path + [action]  # Append the current action to the path
                dataStructure.push((successor, new_path))  # Push the successor and its path to the stack

    return []  # Return an empty list if no path is found

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    dataStructure = util.PriorityQueue()  # Initialize a stack to keep track of nodes
    visited = set()  # Keep track of visited nodes

    # Push initial state into stack along with an empty path and cost
    dataStructure.push((problem.getStartState(), []),0)

    while not dataStructure.isEmpty():
        # Pop a node from the stack
        node, path = dataStructure.pop()

        # Check if the node is the goal state
        if problem.isGoalState(node):
            return path  # Return the path if goal is reached

        # If the node is not visited
        if node not in visited:
            visited.add(node)  # Mark the node as visited

            # Expand the current node and push its children into the stack
            for successor, action, cost in problem.getSuccessors(node):
                new_path = path + [action]  # Append the current action to the path
                g_cost = problem.getCostOfActions(new_path)  # Total cost from the start state to the current successor
                dataStructure.push((successor, new_path), g_cost)  # Push the successor and its path to the stack

    return []  # Return an empty list if no path is found

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    priority_queue = util.PriorityQueue()  # Initialize a priority queue to keep track of nodes
    visited = set()  # Keep track of visited nodes
    trace = []  # Initialize trace to store actions

    start_state = problem.getStartState()
    priority_queue.push((start_state, []), 0 + heuristic(start_state, problem))  # Push initial state into priority queue along with an empty path and priority

    while not priority_queue.isEmpty():
        # Pop a node from the priority queue
        curr_state, path = priority_queue.pop()

        # Check if the node is the goal state
        if problem.isGoalState(curr_state):
            return path  # Return the path if goal is reached

        # If the node is not visited
        if curr_state not in visited:
            visited.add(curr_state)  # Mark the node as visited

            # Expand the current node and push its children into the priority queue
            for successor, action, cost in problem.getSuccessors(curr_state):
                if successor not in visited:
                    new_path = path + [action]  # Append the current action to the path
                    g_cost = problem.getCostOfActions(new_path)  # Total cost from the start state to the current successor
                    h_cost = heuristic(successor, problem)  # Estimated cost from the current successor to the goal state
                    priority_queue.push((successor, new_path), g_cost + h_cost)  # Push the successor and its path to the priority queue with its total cost
                    trace.append(action)  # Record the action taken to reach the successor

    return []  # Return an empty list if no path is found


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
