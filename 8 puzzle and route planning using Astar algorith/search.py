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
import sys
import heapq
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
       
        ##util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """

        ##util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        ##util.raiseNotDefined()

    def getCostOfActions(self, actions):

        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """

        ##util.raiseNotDefined()

    def getHeuristic(self,state):
        """
         state: the current state of agent

         THis function returns the heuristic of current state of the agent which will be the 
         estimated distance from goal.
        """
        ##util.raiseNotDefined()


def aStarSearch(problem):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    pq=util.PriorityQueue()
    closed=set()
    pq.push((problem.getStartState() ,[],0),0)
    while not pq.isEmpty():
        cstate,cmove,ccost=pq.pop()
        if (cstate not in closed):
            closed.add(cstate)
            if problem.isGoalState(cstate):
                return cmove, len(cmove)
        newlst=problem.getSuccessors(cstate)
        print(newlst)
        for i,j,k in newlst:

            h=problem.getHeuristic(i)
            pq.push((i,cmove+[j],ccost+k),ccost+k+h)

    return []
            # cost=problem.costOfActions(i) + problem.getHeuristics(i)
            # if i in pq.heap and cost<problem.costOfActions(i):
            #     pq.pop()
            # elif i in closed.list and cost < problem.costOfActions(i):
            #     closed.pop()
            # elif i not in pq.heap and i not in closed.list:
            #     cost=problem.costOfActions(i)
            #     priority=problem.costOfActions(i) +  problem.getHeuristics(i)
            #     pq.push(pq,i, priority)

