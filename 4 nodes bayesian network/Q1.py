# simple binary tree
# in this implementation, a node is inserted between an existing node and the root
import collections
from itertools import cycle
arg=[]
class BinaryTree():##Here we make a binary tree like structure with all the necessary functions that will be required for us to maintain and perform actions on our tree structures.

    def __init__(self,label,parent,nodetparent=None,nodefparent=None,nodetrue=None,nodefalse=None):
      self.left = None
      self.right = None
      self.label = label
      self.parent=parent
      self.nodetparent=nodetparent
      self.nodefparent=nodefparent
      self.nodetrue=nodetrue
      self.evidence=False
      self.extraval=None
      if self.nodetrue!=None:
              self.nodefalse=1-self.nodetrue
      else:
          self.nodefalse=nodefalse

    def getLeftChild(self):
        return self.left

    def getParent(self):
        return self.parent

    def setEvidence(self,evidence):
        self.evidence=evidence

    def getExtraVal(self):
        return self.extraval
    def setExtraVal(self,val):
        self.extraval=val

    def getEvidence(self):
        return self.evidence

    def getNodetparent(self):
        return self.nodetparent

    def getNodefparent(self):
        return self.nodefparent

    def getNodetrue(self):
        return self.nodetrue

    def getNodefalse(self):
        return self.nodefalse

    def setNodetrue(self,nodetrue):
        self.nodetrue=nodetrue

    def setNodefalse(self,nodefalse):
        self.nodefalse=nodefalse

    def getNabors(self):
        return [self.left,self.right]

    def getRightChild(self):
        return self.right

    def getNodeLabel(self):
        if self.parent=='null':
            return ["Node Label: "+self.label,"Node Parent: "+self.parent,"Node given parent: "+str(self.nodetparent),"Node given not of parent: "+str(self.nodefparent),"Node true prob: "+str(self.nodetrue),"Node false prob: "+str(self.nodefalse),"Evidence: "+str(self.evidence)]
        else:
            return ["Node Label: "+self.label,"Node Parent: "+self.parent.label,"Node given parent: "+str(self.nodetparent),"Node given not of parent: "+str(self.nodefparent),"Node true prob: "+str(self.nodetrue),"Node false prob: "+str(self.nodefalse),"Evidence: "+str(self.evidence)]

    def getJustLabel(self):
        return self.label

    def insertRight(self,newNode,parent,nodetparent,nodefparent,nodetrue,nodefalse):
        if self.right == None:
            self.right = BinaryTree(newNode,parent,nodetparent,nodefparent,nodetrue,nodefalse)


    def insertLeft(self,newNode,parent,nodetparent,nodefparent,nodetrue,nodefalse):
        if self.left == None:
            self.left =BinaryTree(newNode,parent,nodetparent,nodefparent,nodetrue,nodefalse)

def printTree(tree):
        if tree!= None:
            printTree(tree.getLeftChild())
            print(tree.getNodeLabel())
            printTree(tree.getRightChild())

def rootSet():##This function will use our text file and extract the root node from it as well as other arguments from the file
    f = open("text.txt", "r")
    new=f.read()
    lst=new.split("\n")
    for i in range(1,len(lst)):
        arg.append(lst[i].split(","))
    for i in range(len(arg)):
        for j in range(len(arg[i])):
            if arg[i][j]=='null':
                myTree=BinaryTree(arg[i][j-1],'null',None,None,float(arg[i][j+1]))
    return myTree

def PriorProb():#This function will take in the arguments and use the root node function as well to make a tree after adding each node and their conditional probabilties
    root=rootSet()
    lst=[]
    for i in range(1,len(arg)):
        if arg[i][1]==root.getJustLabel():
            if root.getLeftChild()==None:
                root.insertLeft(arg[i][0],root,float(arg[i][2]),float(arg[i][3]),None,None)
            else:
                root.insertRight(arg[i][0],root,float(arg[i][2]),float(arg[i][3]),None,None)
        else:
            visited, queue = set(), collections.deque([root])
            visited.add(root.getJustLabel())
            while queue:
                newroot = queue.popleft()
                vertex=newroot.getNabors()
                for neighbour in vertex:
                    if neighbour!=None:
                        if neighbour.getJustLabel() not in visited:
                            if arg[i][1]==neighbour.getJustLabel():
                                if neighbour.getLeftChild()==None:
                                    neighbour.insertLeft(arg[i][0],neighbour,float(arg[i][2]),float(arg[i][3]),None,None)
                                else:
                                    neighbour.insertRight(arg[i][0],neighbour,float(arg[i][2]),float(arg[i][3]),None,None)
                            visited.add(neighbour.getJustLabel())
                            queue.append(neighbour)

    return root

def printMarginal(tree):#this node traverses the tree and computes marginal probabilty of each node
    visited, queue = set(), collections.deque([tree])
    visited.add(tree.getJustLabel())
    while queue:
        newroot = queue.popleft()
        vertex=newroot.getNabors()
        for neighbour in vertex:
            if neighbour!=None:
                if neighbour.getJustLabel() not in visited:
                    a=neighbour.getNodetparent()*neighbour.getParent().getNodetrue()
                    b=neighbour.getNodefparent()*neighbour.getParent().getNodefalse()
                    neighbour.setNodetrue(a+b)
                    neighbour.setNodefalse(1-neighbour.getNodetrue())
                    visited.add(neighbour.getJustLabel())
                    queue.append(neighbour)
    return tree

def makeInference(tree,label):#This function will first set the evidence true of the node given
    lst=[]
    if tree.getJustLabel()==label:
        tree.setEvidence(True)
        #tree.setEvidence(True)
        tree.setExtraVal(tree.getNodetrue())
        tree.setNodetrue(1)
        tree.setNodefalse(0)
    lst.append(tree)
    visited, queue = set(), collections.deque([tree])
    visited.add(tree.getJustLabel())
    while queue:
        newroot = queue.popleft()
        vertex=newroot.getNabors()
        for neighbour in vertex:
            if neighbour!=None:
                if neighbour.getJustLabel() not in visited:
                    if neighbour.getJustLabel()==label:
                        neighbour.setEvidence(True)
                        neighbour.setExtraVal(neighbour.getNodetrue())
                        neighbour.setNodetrue(1)
                        neighbour.setNodefalse(0)

                    lst.append(neighbour)
                    visited.add(neighbour.getJustLabel())
                    queue.append(neighbour)

    for j in range(len(lst)):#we saved all the nodes that were traversed in the list.
        #print(lst[j].getJustLabel())
        if lst[j].getEvidence()==True:
            for k in range(j-1,0-1,-1):#first we will get the index of the node which has evidence. We will go up from that node changing the posterior priorites of the parent nodes and the parent of the parent nodes and so on


                if lst[k].getParent()!='null':
                    if lst[k+1].getJustLabel()==lst[k].getParent().getJustLabel():
                        if lst[k+1].getEvidence()==True:
                            a=(lst[k+1].getNodetparent()*lst[k].getNodetrue())/lst[k].getNodetrue()
                            lst[k].setNodetrue(a*1+(1-a)*0)
                            lst[k].setNodefalse(1-lst[k].getNodetrue())
                        else:
                            lst[k].setNodetrue(a*lst[k+1].getExtraVal()+(1-a)*(1-lst[k+1].getExtraVal()))
                            lst[k].setNodefalse(1-lst[k].getNodetrue())
                    #print(len(lst))
                if k+1!=len(lst)-1:
                    if lst[k].getParent()!='null':
                        if lst[k+2].getJustLabel()==lst[k].getParent().getJustLabel():
                            if lst[k+2].getEvidence()==True:
                                b=(lst[k+2].getNodetparent()*lst[k].getNodetrue())/lst[k].getNodetrue()
                                lst[k].setNodetrue(b*1+(1-b)*0)
                                lst[k].setNodefalse(1-lst[k].getNodetrue())
                            else:
                                lst[k].setNodetrue(b*lst[k+2].getExtraVal()+(1-b)*(1-lst[k+2].getExtraVal()))
                                lst[k].setNodefalse(1-lst[k].getNodetrue())




            for k in range(j+1,len(lst)):#similarly here we will move downwards from the node that has evidence and calculate posterior prob of its children and the children of its children and so on
                if lst[k-1].getJustLabel()==lst[k].getParent().getJustLabel():

                    if lst[k-1].getEvidence()==True:

                        lst[k].setNodetrue(lst[k].getNodetparent()*1+lst[k].getNodefparent()*0)

                        lst[k].setNodefalse(1-lst[k].getNodetrue())

                    else:
                        lst[k].setNodetrue(lst[k].getNodetparent()*lst[k].getNodetrue()+lst[k].getNodefparent()*lst[k].getNodefalse())

                        lst[k].setNodefalse(1-lst[k].getNodetrue())


                if lst[k-2].getJustLabel()==lst[k].getParent().getJustLabel():
                    if lst[k-2].getEvidence()==True:

                        lst[k].setNodetrue(lst[k].getNodetparent()*1+lst[k].getNodefparent()*0)

                        lst[k].setNodefalse(1-lst[k].getNodetrue())
                    else:
                        lst[k].setNodetrue(lst[k].getNodetparent()*lst[k].getNodetrue()+lst[k].getNodefparent()*lst[k].getNodefalse())

                        lst[k].setNodefalse(1-lst[k].getNodetrue())
                else:
                    lst[k].setNodetrue(lst[k].getNodetparent()*lst[k].getNodetrue()+lst[k].getNodefparent()*lst[k].getNodefalse())

                    lst[k].setNodefalse(1-lst[k].getNodetrue())



    return tree

#part a
#printTree(PriorProb())

#part b
#printTree(printMarginal(PriorProb()))

#part c
#printTree(makeInference(printMarginal(PriorProb()),"A"))

