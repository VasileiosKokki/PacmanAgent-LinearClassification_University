# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal legalActions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the nextState game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        legalActions = gameState.getLegalActions(0)
        max_v = float("-inf")
        maxAction = 'Stop'
        for action in legalActions:
            nextState = gameState.generateSuccessor(0, action)
            # root is max player, so the next player is min player, this is why we call value with index = 1
            v = self.value(nextState, self.depth, 1)
            if v > max_v:
                maxAction = action
                max_v = v
        return maxAction

    def value(self, gameState, depth, ghostIndex):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        if ghostIndex is 0:
            return self.max_value(gameState, depth)
        else:
            return self.min_value(gameState, depth, ghostIndex)

    # Pacman here with index = 0
    def max_value(self, gameState, depth):
        v = float("-inf")
        legalActions = gameState.getLegalActions(0)
        for action in legalActions:
            nextState = gameState.generateSuccessor(0, action)
            v = max(v, self.value(nextState, depth, 1))
        return v

    # Ghosts here with index >=1
    def min_value(self, gameState, depth, ghostIndex):
        v = float("inf")
        legalActions = gameState.getLegalActions(ghostIndex)
        for action in legalActions:
            nextState = gameState.generateSuccessor(ghostIndex, action)
            if ghostIndex == (gameState.getNumAgents() - 1):
                var1 = depth - 1
                var2 = 0
            else:
                var1 = depth
                var2 = ghostIndex + 1
            v = min(v, self.value(nextState, var1, var2))
        return v





class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        legalActions = gameState.getLegalActions(0)
        max_v = float("-inf")
        a = float("-inf")
        b = float("inf")
        maxAction = 'Stop'
        for action in legalActions:
            nextState = gameState.generateSuccessor(0, action)
            # root is max player, so the next player is min player, this is why we call value with index = 1
            v = self.value(nextState, self.depth, 1, a, b)
            a = max(a, v)
            if v > max_v:
                maxAction = action
                max_v = v
        return maxAction

    def value(self, gameState, depth, ghostIndex, a, b):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        if ghostIndex is 0:
            return self.max_value(gameState, depth, a, b)
        else:
            return self.min_value(gameState, depth, ghostIndex, a, b)

    # Pacman here with index = 0
    def max_value(self, gameState, depth, a, b):
        v = float("-inf")
        legalActions = gameState.getLegalActions(0)
        for action in legalActions:
            nextState = gameState.generateSuccessor(0, action)
            v = max(v, self.value(nextState, depth, 1, a, b))
            a = max(a, v)
            if v > b:
                break  # kladema, apla de sinexizoume na eksereunoume pio kato
        return v

    # Ghosts here with index >=1
    def min_value(self, gameState, depth, ghostIndex, a, b):
        v = float("inf")
        legalActions = gameState.getLegalActions(ghostIndex)
        for action in legalActions:
            nextState = gameState.generateSuccessor(ghostIndex, action)
            if ghostIndex == (gameState.getNumAgents() - 1):
                var1 = depth - 1
                var2 = 0
            else:
                var1 = depth
                var2 = ghostIndex + 1
            v = min(v, self.value(nextState, var1, var2, a, b))
            b = min(b, v)
            if v < a:
                break  # kladema, apla de sinexizoume na eksereunoume pio kato
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        legalActions = gameState.getLegalActions(0)
        max_v = float("-inf")
        maxAction = 'Stop'
        for action in legalActions:
            nextState = gameState.generateSuccessor(0, action)
            # root is max player, so the next player is expect player, this is why we call value with index = 1
            v = self.value(nextState, self.depth, 1)
            if v > max_v:
                maxAction = action
                max_v = v
        return maxAction

    def value(self, gameState, depth, ghostIndex):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        if ghostIndex is 0:
            return self.max_value(gameState, depth)
        else:
            return self.expect_value(gameState, depth, ghostIndex)

    # Pacman here with index = 0
    def max_value(self, gameState, depth):
        v = float("-inf")
        legalActions = gameState.getLegalActions(0)
        for action in legalActions:
            nextState = gameState.generateSuccessor(0, action)
            v = max(v, self.value(nextState, depth, 1))
        return v

    # Ghosts here with index >=1
    def expect_value(self, gameState, depth, ghostIndex):
        v = 0
        legalActions = gameState.getLegalActions(ghostIndex)
        numActions = len(legalActions)
        probability = 1.0 / numActions  # Uniform probability for each action
        for action in legalActions:
            nextState = gameState.generateSuccessor(ghostIndex, action)
            if ghostIndex == (gameState.getNumAgents() - 1):
                var1 = depth - 1
                var2 = 0
            else:
                var1 = depth
                var2 = ghostIndex + 1
            v += probability * self.value(nextState, var1, var2)
        return v




def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    pacmanPosition = currentGameState.getPacmanPosition()
    foodPositions = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    sumScaredTimes = sum(ScaredTimes)
    remainingFoods = len(foodPositions)
    remainingCaps = len(currentGameState.getCapsules())
    remainingFoodsMultiplier = 1000000
    foodDistanceMultiplier = 1000
    remainingCapsMultiplier = 1000000
    ghostDistanceMultiplier = 1000


    # Calculate closest food distance
    closestFoodDistance = float("inf")
    for foodPos in foodPositions:
        distance = manhattanDistance(pacmanPosition, foodPos)
        if distance < closestFoodDistance:
            closestFoodDistance = distance

    closestGhostDistance = float("inf")
    closestScaredTime = 0
    # print("\n")
    for ghost in ghostStates:
        # print(ghost, end=" ")
        ghostDistance = manhattanDistance(pacmanPosition, ghost.getPosition())
        if (ghostDistance < 2):  # if ghost is too close, get away as fast as possible
            return float("-inf")
        if ghostDistance < closestGhostDistance:
            closestGhostDistance = ghostDistance
            closestScaredTime = ghost.scaredTimer

    if closestScaredTime == 0:
        score = 1 / (remainingFoods + 1) * remainingFoodsMultiplier - closestGhostDistance + 1 / (closestFoodDistance + 1) * foodDistanceMultiplier + 1 / (remainingCaps + 1) * remainingCapsMultiplier
    else:
        score = 1 / (remainingFoods + 1) * remainingFoodsMultiplier - closestGhostDistance * ghostDistanceMultiplier + 1 / (closestFoodDistance + 1) * foodDistanceMultiplier + 1 / (remainingCaps + 1) * remainingCapsMultiplier
    return score

    # -------------------------- AVERAGE SCORE = 1315.3 (try to beat me) -----------------------------------

# Abbreviation
better = betterEvaluationFunction
