# myTeam.py
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

POS_INF = 999999999
NEG_INF = -999999999
MAX_DEPTH = 3 #depth for the alpha-beta pruning agent

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveABPAgent', second = 'DefensiveABPAgent', numTraining = 0):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)

class OffensiveABPAgent(CaptureAgent):
  """
  For my agent, we will be using an Alpha beta pruning agent
  It will be using minimax search in order to calculate its moves.
  This will mean that we expect the other team to play optimally and so will we. 
  However, there is a high chance that the other team would not play optimally and thus we
  could use expectimax too.
  """
  
  def registerInitialState(self, gameState):
    self.captureLine = [] 
    halfWidth = self.getFood(gameState).width//2
    height = self.getFood(gameState).height

    if self.red:
      for i in range(height):
        if not gameState.hasWall(halfWidth, i):
          self.captureLine.append((halfWidth, i))
    else:
      for i in range(height):
        if not gameState.hasWall(halfWidth-1, i):
          self.captureLine.append((halfWidth-1, i))
    
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    # print("Agent", self.index, "choosing")
    team = self.getTeam(gameState)
    enemy = self.getOpponents(gameState)
    value, action = self.maxVal(gameState, enemy, 0, self.index, NEG_INF, POS_INF)
    
    return action

    util.raiseNotDefined()

  def maxVal(self, gameState, enemy, depth, agentIndex, alpha, beta):
    if gameState.isOver() or depth == MAX_DEPTH:
      return self.evaluationFunction(gameState), None

    for enemyIndex in self.getOpponents(gameState):
      if gameState.getAgentPosition(enemyIndex) == gameState.getAgentPosition(self.index):
        return self.evaluationFunction(gameState), None

    v = NEG_INF
    a = None

    for action in gameState.getLegalActions(agentIndex):
      successor_gameState = gameState.generateSuccessor(agentIndex, action)
      #For now we won't consider the teammate's move since one is focussed on defense.
      v2, a2 = self.minVal(successor_gameState, self.getOpponents(gameState), depth, enemy[0], alpha, beta)
      
      if v2 > v:
        v, a = v2, action
        alpha = max(alpha, v)
      if v > beta:
        return v, a
    return v, a


  def minVal(self, gameState, enemy, depth, agentIndex, alpha, beta):
    #Assume that the minval is our opponent and they want to get to us.
    if gameState.isOver() or depth == MAX_DEPTH or gameState.getAgentPosition(agentIndex) == gameState.getAgentPosition(self.index):
      return self.evaluationFunction(gameState), None

    v = POS_INF
    a = None
    # print("Min")
    # print("Agent Index: ", agentIndex)
    # print("Team:", team)
    # print("Enemy before: ", enemy)
    enemy.remove(agentIndex)
    if(gameState.getAgentPosition(agentIndex) != None):
      # print("Enemy position:", gameState.getAgentPosition(agentIndex))
      for action in gameState.getLegalActions(agentIndex):
        successor_gameState = gameState.generateSuccessor(agentIndex, action)
        if len(enemy) == 0:
          #revert back to the agent performing ABP minimax
          v2, a2 = self.maxVal(successor_gameState, self.getOpponents(gameState), depth+1, self.index, alpha, beta)
        else:
          v2, a2 = self.minVal(successor_gameState, enemy, depth, enemy[0], alpha, beta)

        if v2 < v:
          v, a = v2, action
          beta = min(beta, v)
        if v < alpha:
          return v, action
    elif len(enemy) > 0:
      v2, a2 = self.minVal(gameState, enemy, depth, enemy[0], alpha, beta)
      if v2 < v:
        v = v2
        beta = min(beta, v)
      if(v < alpha):
        return v, a
    else:
      v2, a2 = self.maxVal(gameState, self.getOpponents(gameState), depth+1, self.index, alpha, beta)
      if v2 < v:
        v = v2 
        beta = min(beta, v)
      if(v < alpha):
        return v, a
    return v, a

  def evaluationFunction(self, gameState):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState)
    weights = self.getWeights()

    # for feature in features.keys():
    #   print(feature, features[feature]*weights[feature])
    return features * weights

  def getFeatures(self, gameState):
    features = util.Counter()
    foodList = self.getFood(gameState).asList()
    curPos = gameState.getAgentState(self.index).getPosition()
    enemies = self.getOpponents(gameState)

    enemyDist = []
    for enemy in enemies:
      if gameState.getAgentState(enemy).getPosition() != None:
        if gameState.getAgentState(self.index).isPacman:
          #weighted less for priority of food
          enemyDist.append(self.getMazeDistance(curPos, gameState.getAgentState(enemy).getPosition()))
        else:
          ##weighted more for occassional defending
          enemyDist.append(-self.getMazeDistance(curPos, gameState.getAgentState(enemy).getPosition()))
    
    if(enemyDist):
      features['agentDistfromEnemy'] = min(enemyDist)
      if features['agentDistfromEnemy'] <= 1 and gameState.getAgentState(self.index).isPacman:
        features['danger'] = 1
      features['distFromCaptureLine'] = min([self.getMazeDistance(curPos, capturePos) for capturePos in self.captureLine])
    else:
      features['agentDistfromEnemy'] = 5
    
    features['foodHolding'] = gameState.getAgentState(self.index).numCarrying
    features['foodReturned'] = gameState.getAgentState(self.index).numReturned
    
    if(foodList):
      features['minFoodDist'] = min([self.getMazeDistance(curPos, food) for food in foodList])
    
    features['foodHolding'] = gameState.getAgentState(self.index).numCarrying
    features['foodReturned'] = gameState.getAgentState(self.index).numReturned

    return features

  def getWeights(self):
    return {
      'danger': -1000,
      'agentDistfromEnemy': 3,
      'minFoodDist': -2,
      'foodHolding': 20,
      'foodReturned': 60,
      'distFromCaptureLine':-3
    }
    util.raiseNotDefined()

class DefensiveABPAgent(CaptureAgent):
  """
  For my agent, we will be using an Alpha beta pruning agent
  It will be using minimax search in order to calculate its moves.
  This will mean that we expect the other team to play optimally and so will we. 
  However, there is a high chance that the other team would not play optimally and thus we
  could use expectimax too.
  """
  
  def registerInitialState(self, gameState):
    self.captureLine = [] 
    self.lastEaten = []
    halfWidth = self.getFood(gameState).width//2
    height = self.getFood(gameState).height

    if self.red:
      for i in range(height):
        if not gameState.hasWall(halfWidth, i):
          self.captureLine.append((halfWidth, i))
    else:
      for i in range(height):
        if not gameState.hasWall(halfWidth-1, i):
          self.captureLine.append((halfWidth-1, i))
    
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    # print("Agent", self.index, "choosing")
    team = self.getTeam(gameState)
    enemy = self.getOpponents(gameState)
    value, action = self.maxVal(gameState, enemy, 0, self.index, NEG_INF, POS_INF)
    
    return action

    util.raiseNotDefined()

  def maxVal(self, gameState, enemy, depth, agentIndex, alpha, beta):
    if gameState.isOver() or depth == MAX_DEPTH:
      return self.evaluationFunction(gameState), None

    for enemyIndex in self.getOpponents(gameState):
      if gameState.getAgentPosition(enemyIndex) == gameState.getAgentPosition(self.index):
        return self.evaluationFunction(gameState), None

    v = NEG_INF
    a = None

    for action in gameState.getLegalActions(agentIndex):
      successor_gameState = gameState.generateSuccessor(agentIndex, action)
      #For now we won't consider the teammate's move since one is focussed on defense.
      v2, a2 = self.minVal(successor_gameState, self.getOpponents(gameState), depth, enemy[0], alpha, beta)
      
      if v2 > v:
        v, a = v2, action
        alpha = max(alpha, v)
      if v > beta:
        return v, a
    return v, a


  def minVal(self, gameState, enemy, depth, agentIndex, alpha, beta):
    #Assume that the minval is our opponent and they want to get to us.
    if gameState.isOver() or depth == MAX_DEPTH or gameState.getAgentPosition(agentIndex) == gameState.getAgentPosition(self.index):
      return self.evaluationFunction(gameState), None

    v = POS_INF
    a = None
    # print("Min")
    # print("Agent Index: ", agentIndex)
    # print("Team:", team)
    # print("Enemy before: ", enemy)
    enemy.remove(agentIndex)
    if(gameState.getAgentPosition(agentIndex) != None):
      # print("Enemy position:", gameState.getAgentPosition(agentIndex))
      for action in gameState.getLegalActions(agentIndex):
        successor_gameState = gameState.generateSuccessor(agentIndex, action)
        if len(enemy) == 0:
          #revert back to the agent performing ABP minimax
          v2, a2 = self.maxVal(successor_gameState, self.getOpponents(gameState), depth+1, self.index, alpha, beta)
        else:
          v2, a2 = self.minVal(successor_gameState, enemy, depth, enemy[0], alpha, beta)

        if v2 < v:
          v, a = v2, action
          beta = min(beta, v)
        if v < alpha:
          return v, action
    elif len(enemy) > 0:
      v2, a2 = self.minVal(gameState, enemy, depth, enemy[0], alpha, beta)
      if v2 < v:
        v = v2
        beta = min(beta, v)
      if(v < alpha):
        return v, a
    else:
      v2, a2 = self.maxVal(gameState, self.getOpponents(gameState), depth+1, self.index, alpha, beta)
      if v2 < v:
        v = v2 
        beta = min(beta, v)
      if(v < alpha):
        return v, a
    return v, a

  def evaluationFunction(self, gameState):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState)
    weights = self.getWeights()

    # for feature in features.keys():
    #   print(feature, features[feature]*weights[feature])
    return features * weights

  def getFeatures(self, gameState):
    features = util.Counter()
    selfAgentState = gameState.getAgentState(self.index)
    curPos = selfAgentState.getPosition()

    foodList = self.getFoodYouAreDefending(gameState).asList()

    if(self.getPreviousObservation() != None):
      previousFood = self.getFoodYouAreDefending(self.getPreviousObservation()).asList()
      
      if(len(foodList) < len(previousFood)):
        self.lastEaten = list(set(previousFood)-set(foodList))

      #error check incase there are no elements in food last eaten
      if(self.lastEaten):
        distFromLastFood = min(self.getMazeDistance(curPos, food) for food in self.lastEaten)
        features['distFromLastFood'] = distFromLastFood

    
    enemies = self.getOpponents(gameState)

    enemyDist = []
    for enemy in enemies:
      enemyState = gameState.getAgentState(enemy)
      if enemyState.getPosition() != None and enemyState.isPacman:
        features['enemyOnOurSide'] = 1
        enemyDist.append(self.getMazeDistance(curPos, enemyState.getPosition()))
    
    if(enemyDist):
      #Drop enemy hunting, you found them so we prioritize chasing them.
      features['distFromLastFood'] = 0
      #
      features['agentDistfromEnemy'] = min(enemyDist)
      if features['agentDistfromEnemy'] <= 1 and selfAgentState.scaredTimer > 0:
        features['danger'] = 1  

    return features

  def getWeights(self):
    return {
      'enemyOnOurSide': -1000,
      'distFromLastFood': -2,
      'agentDistFromEnemy': -5,
      'danger': -1000
    }
    util.raiseNotDefined()