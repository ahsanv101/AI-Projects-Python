import search
import random
import csv
import sys

def newHeu(city1,city2):
    cities=[]
    f = open('cities.csv','rb')
    reader = csv.reader(f)
    for row in reader:
        cities.append(row[0])
    f.close()
    index=0
    for i in range(len(cities)):
        if city1==cities[i]:
            index=i+1
        if city2==cities[i]:
            index2=i+1
    connections=[]
    f = open('heuristics.csv','rb')
    reader = csv.reader(f)
    for row2 in reader:
        connections.append(row2[index])
    f.close()
    return connections[index2]
def newConn(city1,city2):
    cities=[]
    f = open('cities.csv','rb')
    reader = csv.reader(f)
    for row in reader:
        cities.append(row[0])
    f.close()
    index=0
    for i in range(len(cities)):
        if city1==cities[i]:
            index=i+1
        if city2==cities[i]:
            index2=i+1
    connections=[]
    f = open('Connections.csv','rb')
    reader = csv.reader(f)
    for row2 in reader:
        connections.append(row2[index])
    f.close()
    return connections[index2]


class RouteState(search.SearchProblem):
    def __init__( self, city1,city2):
        self.currentCity=city1
        self.endCity=city2
        # self.hue=hue
        # self.cost=cost

    def getStartState(self):
        return self.currentCity

    def isGoalState(self,state):
        if state==self.endCity:
            return True
        else:
            return False





    """
          Returns list of (successor, action, stepCost) pairs where
          each succesor is either left, right, up, or down
          from the original state and the cost is 1.0 for each
        """
    def getSuccessors(self,state):
        succ=[]
        #print(state)
        city=state
        cities=[]
        f = open('cities.csv','rb')
        reader = csv.reader(f)
        for row in reader:
            cities.append(row[0])
        f.close()
        for i in cities:
            if newConn(city,i)!= '-1' and newConn(city,i)!= '0':
                succ.append((i,(city,i),int(newConn(city,i))))

        return succ

    def getCostOfActions(self, actions):
        cost=0
        for i in actions:
            cost=cost+i[2]
            print(cost)
        return cost+self.getHeuristic(self.getStartState())


    def getHeuristic(self,state):
        city1=state
        city2=self.endCity
        return int(newHeu(city1,city2))


a=RouteState("Gilgit","Murree")
path = search.aStarSearch(RouteState("Islamabad","Malam Jabba"))
print(path)
#will return a tuple with the list of actions to take ie. [('Islamabad', 'Taxila'), ('Taxila', 'Malam Jabba')] means we will go from islamabad to Taxila and then from taxila to malam jabba and also the steps we took (len of action)
