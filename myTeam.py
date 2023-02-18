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
from game import Directions, Actions
import game
from util import nearestPoint
import time

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'GoalAgent', second = 'GoalAgent', numTraining = 0):
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class GoalAgent(CaptureAgent):
  """
  A base class for defensive agents that defends the food on it's side
  """

  def registerInitialState(self, gameState):
    start_time = time.time()
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

    self.change_x = [0,1,0,-1]
    self.change_y = [1,0,-1,0]

    self.plan = []
    self.captureLine = [] 
    self.closestCapturePoints = {}
    self.foodToGet = []
    
    #a knowledge base part that keeps track if an agent is on our side.
    self.opp_on_our_side = {}
    self.opp_last_seen = {}
    self.opp_scaredTimer = {}
    self.lastFoodEaten = None
    self.opp_seen_this_turn = {}

    #initialise the database for each opponent
    for opp in self.getOpponents(gameState):
      self.opp_on_our_side[opp] = False
      self.opp_last_seen[opp] = None
      self.opp_scaredTimer[opp] = 0
      self.opp_seen_this_turn[opp] = False


    #get dimensions of the maze  
    self.width = self.getFood(gameState).width
    self.height = self.getFood(gameState).height
    #This is the x coordinate that lays next to the middle boundary. 
    #The capture side is considered as the side we are on
    self.captureLineX = 0

    self.halfWidth = self.width//2
    self.halfHeight = self.height//2

    self.vectorTowardsOurSide = 0

    #if we are red, we are on the left side.
    if self.red:
      self.captureLineX = self.halfWidth-1
      self.vectorTowardsOurSide = -1
    else:
      self.captureLineX = self.halfWidth
      self.vectorTowardsOurSide = 1
    
    #go through each position on the line. If it's not a wall, add it to the list.
    for i in range(self.height):
      if not gameState.hasWall(self.captureLineX, i):
        self.captureLine.append((self.captureLineX, i))
    self.preProcessClosestCapturePoint(gameState)

    self.post = None
    postSector = self.halfHeight//2
    team = self.getTeam(gameState)
    team.remove(self.index)
    i = 1
    k = 0
    
    if team[0] < self.index:
      postSector += self.halfHeight
      i *= -1

    supposed_post = (self.captureLineX, postSector)
    
    #the post is a subdivision of each half of the y dimension that sits on the capture line.
    while self.post == None:
      for j in range(2):
        if gameState.hasWall(supposed_post[0], supposed_post[1]):
          supposed_post = (self.captureLineX+k, postSector+i)
          i *= -1
        else:
          self.post = supposed_post
          break
      if(self.post != None):
        break
      else:
        i += 1
      
      if i == self.halfWidth//2:
        k += self.vectorTowardsOurSide

    print(self.index, "Post", self.post)
    

    print("Preprocessing time:", time.time()-start_time)
  
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """

    selfState = gameState.getAgentState(self.index)
    selfPos = selfState.getPosition()

    #Before we make a decision analyse the Scenario.
    #Ask, is the enemy before us? Are they dangerous? Where were they last seen if not seen.

    #danger positions only include a range of an area of an opponent that can potentially send you back.
    #that includes:
    #   - ghosts that are not scared
    #   - any pacman.
    # the following positions include in within a unit of distance from a ghost that isn't scared or a pacman while we're scared.
    enemies_seen = False
    danger_pos = []
    opp_dists = []
    for opp in self.getOpponents(gameState):
      opp_pos = gameState.getAgentPosition(opp)
      opp_state = gameState.getAgentState(opp)
      self.opp_scaredTimer[opp] = opp_state.scaredTimer
      if opp_pos != None:
        enemies_seen = True
        opp_dists.append(self.getMazeDistance(selfPos, opp_pos))
        #if the enemy is seen, update our knowledge
        self.opp_last_seen[opp] = opp_pos
        #Update if they are on our side or not.
        if self.red:
          if gameState.isRed(opp_pos):
            self.opp_on_our_side[opp] = True
          else:
            self.opp_on_our_side[opp] = False
        else:
          if gameState.isRed(opp_pos):
            self.opp_on_our_side[opp] = False
          else:
            self.opp_on_our_side[opp] = True
        #Update if they are Potentially dangerous
        if (not opp_state.isPacman and opp_state.scaredTimer == 0) or (opp_state.isPacman):
          danger_pos.append(opp_pos)
          #They are considered dangerous if we are within range of them (within one block of them after our turn).
          for i in range(len(self.change_x)):
            pos = (opp_pos[0] + self.change_x[i], opp_pos[1]+self.change_y[i])
            if pos not in self.captureLine:
              danger_pos.append((opp_pos[0] + self.change_x[i], opp_pos[1]+self.change_y[i]))

    prevObservation = self.getPreviousObservation()
    if(prevObservation != None):
      food_def_prev = self.getFoodYouAreDefending(prevObservation).asList()
      food_def_now = self.getFoodYouAreDefending(gameState).asList()

      if len(food_def_prev) > len(food_def_now):
        self.lastFoodEaten = list(set(food_def_prev) - set(food_def_now))[0]

    #if we are at the start, it means we must have been eaten.
    #I think we should clear our plan and think of a new plan at this point.
    if selfPos == self.start:
      self.plan.clear()

    if self.plan:
      if self.plan[0] == selfPos:
        self.plan.pop(0)
    

    if self.plan:
      foodList = self.getFood(gameState).asList()
      while self.plan and (self.plan[0] not in foodList and self.plan[0] not in self.captureLine):
        self.plan.pop(0)

    #Start of the decision tree. Current design
    """
                        Losing, Tieing or scared?
                           /             \
                         no              yes
                        /                   \
                      plan?              enemy seen?
                    /    \                 /      \
                  yes     no             no         yes
                 /         \             /            \
              follow plan   \          plan            capsule?
                        enemy seen?                    /      \
                            /                         no       yes
                          yes                        /         \
                          /                      capture     capsule
                    on our side?                   point
                    /         \
                  no          yes
                /               \
              post            chase enemy
    """                       

    if(self.getScore(gameState) <= 0 or selfState.scaredTimer > 0):
      #if we see the enemy, we should try and neutralise them, only if they are a threat
      # and continue with the plan.
      #If we cannot neutralise them, we should retreat.
      if enemies_seen and min(self.opp_scaredTimer.values()) == 0 and min(opp_dists) < 5:
        closest_capsule = self.searchForClosestCapsule(gameState, selfPos)
        closest_capture_point = self.getClosestCapturePoint(selfPos)
        if(closest_capsule != None) and self.plan and (self.plan[0] != closest_capsule):
          self.plan.clear()
          pathplan = self.isReachable(gameState, danger_pos, closest_capsule)
          if(pathplan):
            return pathplan[0]
          else:
            danger_path = self.chargeWithNoConsiderationOfDanger(gameState, closest_capsule)
            if(danger_path):
              return danger_path[0]
            else:
              return Directions.STOP
        elif(closest_capsule == None) and self.plan and self.plan[len(self.plan)-1] != closest_capture_point:
          self.plan.clear()
          pathplan = self.isReachable(gameState, danger_pos, closest_capture_point)
          if(pathplan):
            return pathplan[0]
          else:
            danger_path = self.chargeWithNoConsiderationOfDanger(gameState, closest_capture_point)
            if(danger_path):
              return danger_path[0]
            else:
              return Directions.STOP

      #if we're losing or tieing or cannot defend, try to get points
      unreachable = []
      #if we don't have a plan
      while not self.plan:
        if len(unreachable) > len(self.getFood(gameState).asList()):
          return Directions.STOP
        print("Unreachable:", len(unreachable), unreachable)

        #make a plan
        self.plan = self.positionPlanning(gameState, unreachable)

        #When we successfully make a plan
        if self.plan:
          #check if the plan's first step is reachable
          path = self.isReachable(gameState, danger_pos, self.plan[0])
          #if there is a path, return the first step
          if path:
            return path[0]
          #if not, declare the point as unreachable and clear the plan.
          else:
            unreachable.append(self.plan[0])
            self.plan.clear()
      
      if len(self.plan) > 0:
        pathplan = self.isReachable(gameState, danger_pos, self.plan[0])
        if(pathplan):
          return pathplan[0]
        else:
          danger_path = self.chargeWithNoConsiderationOfDanger(gameState, self.plan[0])
          if(danger_path):
            return danger_path[0]
          else:
            return Directions.STOP
    else:
      #If we're winning, try to get to their post and defend.
      if self.plan and self.plan[0] == selfPos:
        self.plan.pop(0)
      if len(self.plan) > 0:
        pathplan = self.isReachable(gameState, danger_pos, self.plan[0])
        if(pathplan):
          return pathplan[0]
        else:
          danger_path = self.chargeWithNoConsiderationOfDanger(gameState, self.plan[0])
          if(danger_path):
            return danger_path[0]
          else:
            return Directions.STOP
      elif enemies_seen and (True in self.opp_on_our_side.values()): # and selfState.scaredTimer == 0: this is already assumed to be true if the first if fails.
        #calculate closest opponent on our side.
        closest_opp_pos = None
        closest_opp_dist = 999
        for opp in self.getOpponents(gameState):
          if self.opp_on_our_side[opp]:
            opp_pos = gameState.getAgentPosition(opp)
            if opp_pos == None:
              opp_pos = self.opp_last_seen[opp]
            opp_dist = self.getMazeDistance(selfPos, opp_pos)
            if closest_opp_dist > opp_dist:
              closest_opp_pos = opp_pos
              closest_opp_dist = opp_dist

        if closest_opp_pos != None:
          pathplan = self.isReachable(gameState, danger_pos, closest_opp_pos)
          if(pathplan):
            return pathplan[0]
          else:
            return Directions.STOP
        else:
          pathplan = self.isReachable(gameState, danger_pos, self.lastFoodEaten)
          if(pathplan):
            return pathplan[0]
          else:
            return Directions.STOP

      else:
        pathplan = self.isReachable(gameState, danger_pos, self.post)
        if(pathplan):
          return pathplan[0]
        else:
          danger_path = self.chargeWithNoConsiderationOfDanger(gameState, self.post)
          if(danger_path):
            return danger_path[0]
          else:
            return Directions.STOP
  
  def splitFood(self, gameState):
    team = self.getTeam(gameState)
    team.remove(self.index)

    food = self.getFood(gameState).asList()
    foodToGet = []

    if team[0] > self.index:
      #get bottom half
      for f in food:
        if f[1] < self.halfHeight:
          foodToGet.append(f)
    else:
      for f in food:
        if f[1] >= self.halfHeight:
          foodToGet.append(f)

    return foodToGet

  def searchForClosestCapsule(self, gameState, cur_pos):
    """
    Params: 
    - gameState: current gameState object
    - cur_pos: the pos of the agent.

    This method will search for the closestCapsule
    and return its position as a tuple

    If there are no pallets, it will just return the closest Capture Position
    """
    capsules = self.getCapsules(gameState)
    closest = None
    for capsule in capsules:
      if closest == None:
        closest = capsule
      elif self.getMazeDistance(closest, cur_pos) > self.getMazeDistance(capsule, cur_pos):
        closest = capsule

    return closest

  def _getClosestCapturePoint(self, currPos):
    """
    Private method: only used in the preprocessing of closestCapture points.
    """
    closestCapturePoint = self.captureLine[0]
    for pos in self.captureLine:
      if self.getMazeDistance(currPos, pos) < self.getMazeDistance(currPos, closestCapturePoint):
        closestCapturePoint = pos
    
    return closestCapturePoint

  def preProcessClosestCapturePoint(self, gameState):
    """
    This method is like the distancer.py file
    It sets up a dictionary of every coordinate and assigns 
    that coordinate the closest capture point
    """
    for i in range(self.width):
      for j in range(self.height):
        if not gameState.hasWall(i,j):
          self.closestCapturePoints[(i,j)] = self._getClosestCapturePoint((i,j))

  def getClosestCapturePoint(self, currPos):
    return self.closestCapturePoints[currPos]

  def positionPlanning(self, gameState, unreachable):
    """
    Params:
    - gameState: current gameState object
    - unreachable: a list of unreachable positions

    Returns:
    [] or E.g. [(3,2), (3,3) (3,4), (captureLineX, 5)]

    This method will be use to devise a plan to get the minimal cost of getting atleast 3 foods
    and then returning to the capture line.

    This planning does not take any assumptions of the enemy positions

    The plan will be a list of positions corresponding to the foods 
    and the final position as a capture point.
    """
    plan = []
    foodToObtain = self.splitFood(gameState)
    foodToObtain = list(set(foodToObtain) - set(unreachable))

    if len(foodToObtain) <= 2:
      foodToObtain = self.getFood(gameState).asList()
      foodToObtain = list(set(foodToObtain) - set(unreachable))

    """
    Here, we will be using Uniform cost search to look for the plan to get 3 foods.
    """
    #initialise the start node
    start = gameState.getAgentPosition(self.index)
    node = (start, (), 0)
    #the frontier is a priority Queue  from util.py
    frontier = util.PriorityQueue()
    frontier.push(node, node[2])
    #just like in project 1, we use a dictionary to keep track of which states have been reached.
    reached = {}
    reached[node[1]] = node

    while not frontier.isEmpty():
      node = frontier.pop()

      """
      When we expand the node, we check if it's the goal state.
      in this case, the goal state is when we formulate a plan that either
      have 3 foods, or the amount of foods within the plan is 2 less than 
      the amount of food left.
      """
      #if the last element in node[1] is in the captureline list of positions
      if node[1] and node[1][len(node[1])-1] in self.captureLine:
        for pos in node[1]:
          plan.append(pos)
        #We can assume self.captureLine is always filled 
        print("Plan:", plan)
        return plan

      #determine if the next part of the plan is to return to the capture line or get more food.
      nextPositions = []
      #if there are 3 foods, the next position to reach is in the capture line.
      if len(node[1]) == 3 or len(node[1]) == (len(foodToObtain)-2):
        nextPositions = self.captureLine
      else:
        nextPositions = list(set(foodToObtain) - set(node[1]))

      for nextPos in nextPositions:
        currPos = nextPos
        posReached = node[1] + (currPos,)
        totalCost = node[2] + self.getMazeDistance(node[0], currPos)

        if(posReached not in reached) or (totalCost < reached[posReached][2]):
          reached[posReached] = (currPos, posReached, totalCost)
          frontier.push(reached[posReached], totalCost)
    return [] #could not devise a food plan

  def isReachable(self, gameState, danger_pos, goalPos):
    """
    Params:
    - gameState: the current gameState object
    - danger_pos: a list of dangerous positions
    - goalPos: a position that we want to reach.

    Return:
    [] or E.g. ['West', 'North', 'North', 'North']

    This method will be return a list of actions towards the goal Position.

    If the goal position is unreachable or the agent is already at the goal position, 
    it will return an empty list.

    Instead of the getLegalActions method, it uses the generateSafeActions method.
    """

    start = gameState.getAgentPosition(self.index)
    #node format (pos, direction from previous node, cost so far, previous node, gameState)
    node = (start, "START", 0, start, gameState)
    frontier = util.PriorityQueue()
    frontier.push(node, self.getMazeDistance(start, goalPos))
    reached = {}
    reached[start] = node

    """
    This method will use A* search with a heuristic evaluation function
    to return the path. If it cannot reach the goalState, it will return an empty list.
    """
    while not frontier.isEmpty():
      node = frontier.pop()
      if node[0] == goalPos:
        returnPath = self.returnPath(reached, node[0])
        print(self.index, returnPath)
        return returnPath

      for action in self.generateSafeActions(node[4], danger_pos):
        succ = node[4].generateSuccessor(self.index, action)
        succPos = succ.getAgentPosition(self.index)
        total_cost = node[2] + 1


        if (succPos not in reached) or (total_cost < reached[succPos][2]):
          reached[succPos] = (succPos, action, total_cost, node[0], succ)
          frontier.push(reached[succPos], total_cost + self.heuristic(succPos, goalPos))
    return []

  def chargeWithNoConsiderationOfDanger(self, gameState, goalPos):
    """
    This method is like isReachable

    Except it doesnt require a list of dangerous positions. Instead it ignore the
    dangerous positions and returns the best path towards a certain goalPosition.
    """
    start = gameState.getAgentPosition(self.index)
    #node format (pos, direction from previous node, cost so far, previous node, gameState)
    node = (start, "START", 0, start, gameState)
    frontier = util.PriorityQueue()
    frontier.push(node, self.getMazeDistance(start, goalPos))
    reached = {}
    reached[start] = node

    """
    Here we will also be using A* search to get to the goal Position.
    """
    while not frontier.isEmpty():
      node = frontier.pop()

      if node[0] == goalPos:
        return self.returnPath(reached, node[0])

      for action in node[4].getLegalActions(self.index):
        succ = node[4].generateSuccessor(self.index, action)
        succPos = succ.getAgentPosition(self.index)
        total_cost = node[2] + 1

        if (succPos not in reached) or (total_cost < reached[succPos][2]):
          reached[succPos] = (succPos, action, total_cost, node[0], succ)
          frontier.push(reached[succPos], total_cost + self.heuristic(succPos, goalPos))
    return []

  def returnPath(self, reached, goalPos):
    """
    Given a dictionary with nodes of format (curPos, action from prevPos, cost, prevPos)
    return a list of actions to get to goal node.
    """
    path = []
    #start from goal node and work its way backwards.
    node = goalPos
    while reached[node][1] != "START":
      path.insert(0, reached[node][1])
      node = reached[node][3]

    return path

  def heuristic(self, pos, goalPos):
    """
    this heuristic function returns the minimal distance between the
    position provided and the goal position.
    """
    return self.getMazeDistance(pos, goalPos)
    util.raiseNotDefined()
    

  def generateSafeActions(self, gameState, danger_pos):
    """
    This is a successor action function for the isReachable method

    It returns a list of safe actions which won't return the agent back to the start.
    """    
    safeActions = []
    legalActions = gameState.getLegalActions(self.index)

    for legalAction in legalActions:
      successor = gameState.generateSuccessor(self.index, legalAction)
      sucState = gameState.getAgentState(self.index)
      sucPos = sucState.getPosition()

      #We are only vulnerable if: 
      # - we are pacman
      # - we are a ghost that is scared

      if not ((sucPos in danger_pos) and 
      ((sucState.scaredTimer > 0 and not sucState.isPacman) or (sucState.isPacman))):
        safeActions.append(legalAction)

    return safeActions


class GoalAgentUpdate(CaptureAgent):
  """
  A base class for defensive agents that defends the food on it's side
  """

  def registerInitialState(self, gameState):

    self.enemyChange = None

    start_time = time.time()
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

    self.change_x = [0,1,0,-1]
    self.change_y = [1,0,-1,0]

    self.plan = []
    self.captureLine = [] 
    self.closestCapturePoints = {}
    self.foodToGet = []
    
    #a knowledge base part that keeps track if an agent is on our side.
    self.opp_on_our_side = {}
    self.opp_last_seen = {}
    self.opp_scaredTimer = {}
    self.lastFoodEaten = None
    self.opp_seen_this_turn = {}

    #initialise the database for each opponent
    for opp in self.getOpponents(gameState):
      self.opp_on_our_side[opp] = False
      self.opp_last_seen[opp] = None
      self.opp_scaredTimer[opp] = 0
      self.opp_seen_this_turn[opp] = False


    #get dimensions of the maze  
    self.width = self.getFood(gameState).width
    self.height = self.getFood(gameState).height
    #This is the x coordinate that lays next to the middle boundary. 
    #The capture side is considered as the side we are on
    self.captureLineX = 0

    self.halfWidth = self.width//2
    self.halfHeight = self.height//2

    self.vectorTowardsOurSide = 0

    #if we are red, we are on the left side.
    if self.red:
      self.captureLineX = self.halfWidth-1
      self.vectorTowardsOurSide = -1
    else:
      self.captureLineX = self.halfWidth
      self.vectorTowardsOurSide = 1
    
    #go through each position on the line. If it's not a wall, add it to the list.
    for i in range(self.height):
      if not gameState.hasWall(self.captureLineX, i):
        self.captureLine.append((self.captureLineX, i))
    self.preProcessClosestCapturePoint(gameState)

    self.post = None
    postSector = self.halfHeight//2
    team = self.getTeam(gameState)
    team.remove(self.index)
    i = 1
    k = 0
    
    if team[0] < self.index:
      postSector += self.halfHeight
      i *= -1

    supposed_post = (self.captureLineX, postSector)
    
    #the post is a subdivision of each half of the y dimension that sits on the capture line.
    while self.post == None:
      for j in range(2):
        if gameState.hasWall(supposed_post[0], supposed_post[1]):
          supposed_post = (self.captureLineX+k, postSector+i)
          i *= -1
        else:
          self.post = supposed_post
          break
      if(self.post != None):
        break
      else:
        i += 1
      
      if i == self.halfWidth//2:
        k += self.vectorTowardsOurSide
    
    
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    selfState = gameState.getAgentState(self.index)
    selfPos = selfState.getPosition()

    #Before we make a decision analyse the Scenario.
    #Ask, is the enemy before us? Are they dangerous? Where were they last seen if not seen.

    #danger positions only include a range of an area of an opponent that can potentially send you back.
    #that includes:
    #   - ghosts that are not scared
    #   - any pacman.
    # the following positions include in within a unit of distance from a ghost that isn't scared or a pacman while we're scared.
    enemies_seen = False
    danger_pos = []
    opp_dists = []
    for opp in self.getOpponents(gameState):
      opp_pos = gameState.getAgentPosition(opp)
      opp_state = gameState.getAgentState(opp)
      self.opp_scaredTimer[opp] = opp_state.scaredTimer
      if opp_pos != None:
        enemies_seen = True
        opp_dists.append(self.getMazeDistance(selfPos, opp_pos))
        #if the enemy is seen, update our knowledge
        self.opp_last_seen[opp] = opp_pos
        #Update if they are on our side or not.
        if self.red:
          if gameState.isRed(opp_pos):
            self.opp_on_our_side[opp] = True
          else:
            self.opp_on_our_side[opp] = False
        else:
          if gameState.isRed(opp_pos):
            self.opp_on_our_side[opp] = False
          else:
            self.opp_on_our_side[opp] = True

        #Update if they are Potentially dangerous
        if (not opp_state.isPacman and opp_state.scaredTimer == 0) or (opp_state.isPacman):
          danger_pos.append(opp_pos)
          #They are considered dangerous if we are within range of them (within one block of them after our turn).
          for i in range(len(self.change_x)):
            danger_pos.append((opp_pos[0] + self.change_x[i], opp_pos[1]+self.change_y[i]))

    prevObservation = self.getPreviousObservation()
    if(prevObservation != None):
      food_def_prev = self.getFoodYouAreDefending(prevObservation).asList()
      food_def_now = self.getFoodYouAreDefending(gameState).asList()

      if len(food_def_prev) > len(food_def_now):
        self.lastFoodEaten = list(set(food_def_prev) - set(food_def_now))[0]

    #if we are at the start, it means we must have been eaten.
    #I think we should clear our plan and think of a new plan at this point.
    if selfPos == self.start:
      self.plan.clear()

    if self.plan:
      if self.plan[0] == selfPos:
        self.plan.pop(0)

    if self.plan:
      foodList = self.getFood(gameState).asList()
      while self.plan and (self.plan[0] not in foodList and self.plan[0] not in self.captureLine):
        self.plan.pop(0)

    #Start of the decision tree. Current design
    """
                        Losing, Tieing or scared?
                           /             \
                         no              yes
                        /                   \
                      plan?              enemy seen?
                    /    \                 /      \
                  yes     no             no         yes
                 /         \             /            \
              follow plan   \          plan         capsule?
                        enemy seen?                    /      \
                            /                      no       yes
                          yes                        /         \
                          /                      capture     capsule
                    on our side?                         point
                    /         \
                  no          yes
                /               \
              post            chase enemy
    """                       

    if(self.getScore(gameState) <= 0 or selfState.scaredTimer > 0):
      
      #if we see the enemy, we should try and neutralise them, only if they are a threat
      # and continue with the plan.
      #If we cannot neutralise them, we should retreat.
      if enemies_seen and not opp_state.isPacman and selfState.isPacman and min(self.opp_scaredTimer.values()) == 0 and min(opp_dists) < 5:
        self.plan.clear()

        closest_point = 9999
        dict_closest_point = {}
        closest = None

        closest_capsule = self.searchForClosestCapsule(gameState, selfPos)
        closest_capture_point = self.getClosestCapturePoint(selfPos)
        
        if closest_capsule:
          dict_closest_point[closest_capsule] = self.getMazeDistance(selfPos, closest_capsule)

        if closest_capture_point:
          dict_closest_point[closest_capture_point] = self.getMazeDistance(selfPos, closest_capture_point)

        closest_point = min(dict_closest_point, key=dict_closest_point.get)

        if(closest_capsule != None):
          closest = closest_point
        else:
          closest = closest_capture_point
          
        pathplan = self.isReachable(gameState, danger_pos, closest)
        if(pathplan):
          return pathplan[0]
        else:
          danger_path = self.chargeWithNoConsiderationOfDanger(gameState, closest)
          if(danger_path):
            return danger_path[0]
          else:
            return Directions.STOP
      
      if (True in self.opp_on_our_side.values() and enemies_seen): # and selfState.scaredTimer == 0: this is already assumed to be true if the first if fails.

        point = None
        enemyToChase = []
        self.plan.clear()
        enemyToChase = self.splitEnemy(gameState,self.enemyChange,danger_pos)
        
        if enemyToChase:
          for opp_pos in enemyToChase:

            if selfPos == opp_pos:
              enemies_seen == False

            if opp_pos != None:
              point = opp_pos
            else:
               point = self.post

            pathplan = self.isReachable(gameState, danger_pos, point)
            if(pathplan):
              return pathplan[0]
            else:
              danger_path = self.chargeWithNoConsiderationOfDanger(gameState, point)
            if danger_path:
              return danger_path[0]

        else:
          pass

      if len(self.plan) > 0:
        pathplan = self.isReachable(gameState, danger_pos, self.plan[0])
        if(pathplan):
          return pathplan[0]
        else:
          danger_path = self.chargeWithNoConsiderationOfDanger(gameState, self.plan[0])
          if(danger_path):
            return danger_path[0]
          else:
            return Directions.STOP

      path = self.foodPlan(gameState,danger_pos)
      return path

    else:
      #If we're winning, try to get to their post and defend.

      if enemies_seen and not opp_state.isPacman and selfState.isPacman and min(self.opp_scaredTimer.values()) == 0 and min(opp_dists) < 5:
        
        closest = None
        closest_point = 9999
        dict_closest_point = {}

        closest_capsule = self.searchForClosestCapsule(gameState, selfPos)
        closest_capture_point = self.getClosestCapturePoint(selfPos)
        
        if closest_capsule:
          dict_closest_point[closest_capsule] = self.getMazeDistance(selfPos, closest_capsule)

        if closest_capture_point:
          dict_closest_point[closest_capture_point] = self.getMazeDistance(selfPos, closest_capture_point)

        closest_point = min(dict_closest_point, key=dict_closest_point.get)

        if(closest_capsule != None):
          closest = closest_point
        else:
          closest = closest_capture_point

        pathplan = self.isReachable(gameState, danger_pos, closest)

        if(pathplan):
          return pathplan[0]
        else:
          danger_path = self.chargeWithNoConsiderationOfDanger(gameState, closest)
          if(danger_path):
            return danger_path[0]
          else:
            return Directions.STOP

      if (True in self.opp_on_our_side.values() and enemies_seen): # and selfState.scaredTimer == 0: this is already assumed to be true if the first if fails.
        enemyToChase = []
        enemyToChase = self.splitEnemy(gameState,self.enemyChange,danger_pos)

        if enemyToChase:
          for opp_pos in enemyToChase:
            if opp_pos != None:
              self.plan.clear()
              pathplan = self.isReachable(gameState, danger_pos, opp_pos)
              if(pathplan):
                return pathplan[0]
              else:
                danger_path = self.chargeWithNoConsiderationOfDanger(gameState, opp_pos)
              if danger_path:
                return danger_path[0]
              else:
                pathplan = self.isReachable(gameState, danger_pos, self.post)
                if(pathplan):
                  return pathplan[0]
                else:
                  danger_path = self.chargeWithNoConsiderationOfDanger(gameState, self.post)
                  if(danger_path):
                    return danger_path[0]
                  else:
                    return Directions.STOP

      if self.plan and self.plan[0] == selfPos:
        self.plan.pop(0)

      point = None

      if len(self.plan) > 0:
        point = self.plan[0]
      else:
        point = self.post        

      pathplan = self.isReachable(gameState, danger_pos, point)
      if(pathplan):
        return pathplan[0]
      else:
        danger_path = self.chargeWithNoConsiderationOfDanger(gameState, point)
        if(danger_path):
          return danger_path[0]
        else:
          return Directions.STOP


  def splitFood(self, gameState):
    team = self.getTeam(gameState)
    team.remove(self.index)

    food = self.getFood(gameState).asList()
    foodToGet = []

    if team[0] > self.index:
      #get bottom half
      for f in food:
        if f[1] < self.halfHeight:
          foodToGet.append(f)
    else:
      for f in food:
        if f[1] >= self.halfHeight:
          foodToGet.append(f)

    return foodToGet

  def searchForClosestCapsule(self, gameState, cur_pos):
    """
    Params: 
    - gameState: current gameState object
    - cur_pos: the pos of the agent.
    This method will search for the closestCapsule
    and return its position as a tuple
    If there are no pallets, it will just return the closest Capture Position
    """
    capsules = self.getCapsules(gameState)
    closest = None
    for capsule in capsules:
      if closest == None:
        closest = capsule
      elif self.getMazeDistance(closest, cur_pos) > self.getMazeDistance(capsule, cur_pos):
        closest = capsule

    return closest

  def _getClosestCapturePoint(self, currPos):
    """
    Private method: only used in the preprocessing of closestCapture points.
    """
    closestCapturePoint = self.captureLine[0]
    for pos in self.captureLine:
      if self.getMazeDistance(currPos, pos) < self.getMazeDistance(currPos, closestCapturePoint):
        closestCapturePoint = pos
    
    return closestCapturePoint

  def preProcessClosestCapturePoint(self, gameState):
    """
    This method is like the distancer.py file
    It sets up a dictionary of every coordinate and assigns 
    that coordinate the closest capture point
    """
    for i in range(self.width):
      for j in range(self.height):
        if not gameState.hasWall(i,j):
          self.closestCapturePoints[(i,j)] = self._getClosestCapturePoint((i,j))

  def getClosestCapturePoint(self, currPos):
    return self.closestCapturePoints[currPos]

  def positionPlanning(self, gameState, unreachable):
    """
    Params:
    - gameState: current gameState object
    - unreachable: a list of unreachable positions
    Returns:
    [] or E.g. [(3,2), (3,3) (3,4), (captureLineX, 5)]
    This method will be use to devise a plan to get the minimal cost of getting atleast 3 foods
    and then returning to the capture line.
    This planning does not take any assumptions of the enemy positions
    The plan will be a list of positions corresponding to the foods 
    and the final position as a capture point.
    """
    plan = []
    foodToObtain = self.splitFood(gameState)
    foodToObtain = list(set(foodToObtain) - set(unreachable))

    if len(foodToObtain) <= 2:
      foodToObtain = self.getFood(gameState).asList()
      foodToObtain = list(set(foodToObtain) - set(unreachable))

    """
    Here, we will be using Uniform cost search to look for the plan to get 3 foods.
    """
    #initialise the start node
    start = gameState.getAgentPosition(self.index)
    node = (start, (), 0)
    #the frontier is a priority Queue  from util.py
    frontier = util.PriorityQueue()
    frontier.push(node, node[2])
    #just like in project 1, we use a dictionary to keep track of which states have been reached.
    reached = {}
    reached[node[1]] = node

    while not frontier.isEmpty():
      node = frontier.pop()

      """
      When we expand the node, we check if it's the goal state.
      in this case, the goal state is when we formulate a plan that either
      have 3 foods, or the amount of foods within the plan is 2 less than 
      the amount of food left.
      """
      if node[1] and node[1][len(node[1])-1] in self.captureLine:
        for pos in node[1]:
          plan.append(pos)

        
        #look for the shortest path to the capture line
        # currPos = node[0]
        #We can assume self.captureLine is always filled 
        # plan.append(self.getClosestCapturePoint(currPos))
        return plan

      #determine if the next part of the plan is to return to the capture line or get more food.
      nextPositions = []
      #if the last element in node[1] is in the captureline list of positions
      if len(node[1]) == 3 or len(node[1]) == (len(foodToObtain)-2):
        nextPositions = self.captureLine
      else:
        nextPositions = list(set(foodToObtain) - set(node[1]))

      for nextPos in nextPositions:
        currPos = nextPos
        posReached = node[1] + (currPos,)
        totalCost = node[2] + self.getMazeDistance(node[0], currPos)

        if(posReached not in reached) or (totalCost < reached[posReached][2]):
          reached[posReached] = (currPos, posReached, totalCost)
          frontier.push(reached[posReached], totalCost)
    return [] #could not devise a food plan

  def isReachable(self, gameState, danger_pos, goalPos):
    """
    Params:
    - gameState: the current gameState object
    - danger_pos: a list of dangerous positions
    - goalPos: a position that we want to reach.
    Return:
    [] or E.g. ['West', 'North', 'North', 'North']
    This method will be return a list of actions towards the goal Position.
    If the goal position is unreachable or the agent is already at the goal position, 
    it will return an empty list.
    Instead of the getLegalActions method, it uses the generateSafeActions method.
    """
    start = gameState.getAgentPosition(self.index)
    #node format (pos, direction from previous node, cost so far, previous node, gameState)
    node = (start, "START", 0, start, gameState)
    frontier = util.PriorityQueue()
    frontier.push(node, self.getMazeDistance(start, goalPos))
    reached = {}
    reached[start] = node

    """
    This method will use A* search with a heuristic evaluation function
    to return the path. If it cannot reach the goalState, it will return an empty list.
    """
    while not frontier.isEmpty():
      node = frontier.pop()
      if node[0] == goalPos:
        returnPath = self.returnPath(reached, node[0])
        return returnPath

      for action in self.generateSafeActions(node[4], danger_pos):
        succ = node[4].generateSuccessor(self.index, action)
        succPos = succ.getAgentPosition(self.index)
        total_cost = node[2] + 1


        if (succPos not in reached) or (total_cost < reached[succPos][2]):
          reached[succPos] = (succPos, action, total_cost, node[0], succ)
          frontier.push(reached[succPos], total_cost + self.heuristic(succPos, goalPos))
    return []

  def chargeWithNoConsiderationOfDanger(self, gameState, goalPos):
    """
    This method is like isReachable
    Except it doesnt require a list of dangerous positions. Instead it ignore the
    dangerous positions and returns the best path towards a certain goalPosition.
    """
    start = gameState.getAgentPosition(self.index)
    #node format (pos, direction from previous node, cost so far, previous node, gameState)
    node = (start, "START", 0, start, gameState)
    frontier = util.PriorityQueue()
    frontier.push(node, self.getMazeDistance(start, goalPos))
    reached = {}
    reached[start] = node

    """
    Here we will also be using A* search to get to the goal Position.
    """
    while not frontier.isEmpty():
      node = frontier.pop()

      if node[0] == goalPos:
        return self.returnPath(reached, node[0])

      for action in node[4].getLegalActions(self.index):
        succ = node[4].generateSuccessor(self.index, action)
        succPos = succ.getAgentPosition(self.index)
        total_cost = node[2] + 1

        if (succPos not in reached) or (total_cost < reached[succPos][2]):
          reached[succPos] = (succPos, action, total_cost, node[0], succ)
          frontier.push(reached[succPos], total_cost + self.heuristic(succPos, goalPos))
    return []

  def returnPath(self, reached, goalPos):
    """
    Given a dictionary with nodes of format (curPos, action from prevPos, cost, prevPos)
    return a list of actions to get to goal node.
    """
    path = []
    #start from goal node and work its way backwards.
    node = goalPos
    while reached[node][1] != "START":
      path.insert(0, reached[node][1])
      node = reached[node][3]
      

    return path

  def heuristic(self, pos, goalPos):
    """
    this heuristic function returns the minimal distance between the
    position provided and the goal position.
    """
    return self.getMazeDistance(pos, goalPos)
    util.raiseNotDefined()
    

  def generateSafeActions(self, gameState, danger_pos):
    """
    This is a successor action function for the isReachable method
    It returns a list of safe actions which won't return the agent back to the start.
    """    
    safeActions = []
    legalActions = gameState.getLegalActions(self.index)

    for legalAction in legalActions:
      successor = gameState.generateSuccessor(self.index, legalAction)
      sucState = gameState.getAgentState(self.index)
      sucPos = sucState.getPosition()

      #We are only vulnerable if: 
      # - we are pacman
      # - we are a ghost that is scared

      if not ((sucPos in danger_pos) and 
      ((sucState.scaredTimer > 0 and not sucState.isPacman) or (sucState.isPacman))):
        safeActions.append(legalAction)
    return safeActions


  def foodPlan(self,gameState,danger_pos):

      #if we're losing or tieing or cannot defend, try to get points
      unreachable = []
      while not self.plan:
        if len(unreachable) > len(self.getFood(gameState).asList()):
          return Directions.STOP

        #make a plan
        self.plan = self.positionPlanning(gameState, unreachable)

        #When we successfully make a plan
        if self.plan:
          #check if the plan's first step is reachable
          path = self.isReachable(gameState, danger_pos, self.plan[0])
          #if there is a path, return the first step
          if path:
            return path[0]
          #if not, declare the point as unreachable and clear the plan.
          else:
            unreachable.append(self.plan[0])
            self.plan.clear()

  def splitEnemy(self, gameState,enemyChange,danger_pos):
    team = self.getTeam(gameState)
    team.remove(self.index)

    enemyToChase = []

    pos = None

    opp_dict = {}

      #get bottom half
    Agent1pos = gameState.getAgentPosition(self.index)
    Agent2pos = gameState.getAgentPosition(team[0])
    
    for opponent in self.getOpponents(gameState):
      if self.opp_on_our_side[opponent] and gameState.getAgentPosition(opponent) != None:
        pos = gameState.getAgentPosition(opponent)
        
      else:
        if self.opp_last_seen[opponent]:
          if enemyChange == None:
            enemyChange = self.opp_last_seen[opponent]
            pos = self.opp_last_seen[opponent]
          else:
            if self.opp_last_seen[opponent] != enemyChange:
              pos = self.opp_last_seen[opponent]
              enemyChange = self.opp_last_seen[opponent]
            else:
              if self.lastFoodEaten:
                pos = self.lastFoodEaten
              else:
                  pass
        else:
          if self.lastFoodEaten:
            pos = self.lastFoodEaten
          else:
            pass
      
      if pos != None:
        opp_dict[Agent1pos] = self.getMazeDistance(Agent1pos, pos)
        opp_dict[Agent2pos] = self.getMazeDistance(Agent2pos, pos)
      else:
        pass

      # if team[0] > self.index:
      if opp_dict[Agent1pos] <  opp_dict[Agent2pos]:
          enemyToChase.append(pos)
      else:
        pass
       
    return enemyToChase
