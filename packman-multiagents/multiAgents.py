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
import sys

from game import Agent
from util import manhattanDistance


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # First we check the successor state.
        # If the successor state was the win state, then we return 
        # the highest value possible.
        if successorGameState.isWin():
            return sys.maxsize

        # Manhattan distance to the available foods from the successor state 
        foodList = newFood.asList()
        foodDistance = [0]
        for pos in foodList:
            foodDistance.append( manhattanDistance(newPos,pos) )
            
        # Manhattan distance to each ghost in the game from successor state
        ghostPos = []
        for ghost in newGhostStates:
            ghostPos.append(ghost.getPosition())

        ghostDistance = []
        for pos in ghostPos:
            ghostDistance.append(manhattanDistance(newPos,pos))

        # Manhattan distance to each ghost in the game from current state
        ghostPosCurrent = []
        for ghost in currentGameState.getGhostStates():
            ghostPosCurrent.append(ghost.getPosition())

        ghostDistanceCurrent = []
        for pos in ghostPosCurrent:
            ghostDistanceCurrent.append(manhattanDistance(newPos,pos))

        # Initializing with the score of 0
        score = 0
        # Get Number of food available in successor state
        numberOfFoodLeft = len(foodList)
        # Get Number of food available in current state
        numberOfFoodLeftCurrent = len(currentGameState.getFood().asList())
        # Get Number of Power Pellets available in successor state
        numberofPowerPellets = len(successorGameState.getCapsules())
        # Get state of ghosts in successor state
        sumScaredTimes = sum(newScaredTimes)
            
        # Relative Score    
        score += successorGameState.getScore() - currentGameState.getScore()
        if action == Directions.STOP:
            #Penalty for stop
            score -= 10
            
        # Add Score if pacman eats power pellet in next state
        if newPos in currentGameState.getCapsules():
            score += 150 * numberofPowerPellets
        # Add score if there are lesser number of food available in successor state
        if numberOfFoodLeft < numberOfFoodLeftCurrent:
            score += 200

        # For each food left subtract 10 score.     
        score -= 10 * numberOfFoodLeft

        # If ghosts are scared lesser distance to ghosts is better
        if sumScaredTimes > 0 :
            if min(ghostDistanceCurrent) < min(ghostDistance):
                score += 200
            else:
                score -=100
        # If ghosts are not scared greater distance to ghosts is better
        else:
            if min(ghostDistanceCurrent) < min(ghostDistance):
                score -= 100
            else:
                score += 200
        
        return score


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        # First we get the number of ghosts in the game
        numberOfGhosts = gameState.getNumAgents() - 1

        # This method is the pacman level
        def maxLevel(gameState,depth):
            currDepth = depth + 1 # Incresing depth
            if gameState.isWin() or gameState.isLose() or currDepth==self.depth: # Termination Test 
                return self.evaluationFunction(gameState)
            maxvalue = -999999 # Starting from the lowest
            actions = gameState.getLegalActions(0)
            for action in actions:
                successor= gameState.generateSuccessor(0,action)
                maxvalue = max (maxvalue,minLevel(successor,currDepth,1))
            return maxvalue # Returning the max value
        
        # This method is the ghosts level
        def minLevel(gameState,depth, agentIndex):
            minvalue = 999999 # Starting from highest
            if gameState.isWin() or gameState.isLose(): # Termination Test 
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(agentIndex)
            for action in actions:
                successor= gameState.generateSuccessor(agentIndex,action)
                if agentIndex == (gameState.getNumAgents() - 1):
                    minvalue = min (minvalue,maxLevel(successor,depth))
                else:
                    minvalue = min(minvalue,minLevel(successor,depth,agentIndex+1))
            return minvalue # Returning the min value
        
        # Get actions
        actions = gameState.getLegalActions(0)
        currentScore = -999999
        returnAction = ''
        for action in actions:
            nextState = gameState.generateSuccessor(0,action)
            # Next level is a min level. Hence calling min for successors of the root
            score = minLevel(nextState,0,1)
            # Choosing the action which is Maximum of the successors
            if score > currentScore:
                returnAction = action
                currentScore = score
        return returnAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # Used only for pacman agent hence agentindex is always 0 
        def maxLevel(gameState,depth,alpha, beta):
            currDepth = depth + 1
            if gameState.isWin() or gameState.isLose() or currDepth==self.depth: 
                return self.evaluationFunction(gameState)
            maxvalue = -999999
            actions = gameState.getLegalActions(0)
            alpha1 = alpha
            for action in actions:
                successor= gameState.generateSuccessor(0,action)
                maxvalue = max (maxvalue,minLevel(successor,currDepth,1,alpha1,beta))
                if maxvalue > beta:
                    return maxvalue
                alpha1 = max(alpha1,maxvalue)
            return maxvalue
        
        # For all ghosts
        def minLevel(gameState,depth,agentIndex,alpha,beta):
            minvalue = 999999
            if gameState.isWin() or gameState.isLose(): 
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(agentIndex)
            beta1 = beta
            for action in actions:
                successor= gameState.generateSuccessor(agentIndex,action)
                if agentIndex == (gameState.getNumAgents()-1):
                    minvalue = min (minvalue,maxLevel(successor,depth,alpha,beta1))
                    if minvalue < alpha:
                        return minvalue
                    beta1 = min(beta1,minvalue)
                else:
                    minvalue = min(minvalue,minLevel(successor,depth,agentIndex+1,alpha,beta1))
                    if minvalue < alpha:
                        return minvalue
                    beta1 = min(beta1,minvalue)
            return minvalue

        # Get actions
        actions = gameState.getLegalActions(0)
        currentScore = -999999
        returnAction = ''
        # Alpha Beta limits
        alpha = -999999
        beta = 999999
        for action in actions:
            nextState = gameState.generateSuccessor(0,action)
            # Next level is a min level. Hence calling min for successors of the root
            score = minLevel(nextState,0,1,alpha,beta)
            # Choosing the action which is Maximum of the successors
            if score > currentScore:
                returnAction = action
                currentScore = score
            # Updating alpha value at root
            if score > beta:
                return returnAction
            alpha = max(alpha,score)
        return returnAction


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
        # Used only for pacman agent hence agentindex is always 0
        def maxLevel(gameState,depth):
            currDepth = depth + 1
            if gameState.isWin() or gameState.isLose() or currDepth==self.depth:   
                return self.evaluationFunction(gameState)
            maxvalue = -999999
            actions = gameState.getLegalActions(0)
            for action in actions:
                successor= gameState.generateSuccessor(0,action)
                maxvalue = max (maxvalue,expectLevel(successor,currDepth,1))
            return maxvalue
        
        # For all ghosts
        def expectLevel(gameState,depth, agentIndex):
            if gameState.isWin() or gameState.isLose():   
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(agentIndex)
            totalexpectedvalue = 0
            numberofactions = len(actions)
            for action in actions:
                successor= gameState.generateSuccessor(agentIndex,action)
                if agentIndex == (gameState.getNumAgents() - 1):
                    expectedvalue = maxLevel(successor,depth)
                else:
                    expectedvalue = expectLevel(successor,depth,agentIndex+1)
                totalexpectedvalue = totalexpectedvalue + expectedvalue
            if numberofactions == 0:
                return  0
            return float(totalexpectedvalue) / float(numberofactions)
        
        # Root level action
        actions = gameState.getLegalActions(0)
        currentScore = -999999
        returnAction = ''
        for action in actions:
            nextState = gameState.generateSuccessor(0,action)
            # Next level is a expect level. Hence calling expectLevel for successors of the root
            score = expectLevel(nextState,0,1)
            # Choosing the action which is Maximum of the successors
            if score > currentScore:
                returnAction = action
                currentScore = score
        return returnAction



def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ ghostState.scaredTimer for ghostState in newGhostStates ]
    
    # Manhattan distance to the foods from the current state
    foodList = newFood.asList()
    from util import manhattanDistance
    foodDistance = [0]
    for pos in foodList:
        foodDistance.append(manhattanDistance(newPos,pos))

    # Manhattan distance to each ghost from the current state
    ghostPos = []
    for ghost in newGhostStates:
        ghostPos.append(ghost.getPosition())
        
    ghostDistance = [0]
    for pos in ghostPos:
        ghostDistance.append(manhattanDistance(newPos,pos))

    numberofPowerPellets = len(currentGameState.getCapsules())

    score = 0
    numberOfNoFoods = len(newFood.asList(False))           
    sumScaredTimes = sum(newScaredTimes)
    sumGhostDistance = sum (ghostDistance)
    reciprocalfoodDistance = 0
    if sum(foodDistance) > 0:
        reciprocalfoodDistance = 1.0 / sum(foodDistance)
        
    score += currentGameState.getScore()  + reciprocalfoodDistance + numberOfNoFoods

    if sumScaredTimes > 0:    
        score +=   sumScaredTimes + (-1 * numberofPowerPellets) + (-1 * sumGhostDistance)
    else :
        score +=  sumGhostDistance + numberofPowerPellets
    return score


# Abbreviation
better = betterEvaluationFunction
