from time import sleep
from math import inf
from random import randint
from random import choice

class ultimateTicTacToe:
    def __init__(self):
        """
        Initialization of the game.
        """
        self.board=[['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_']]
        self.maxPlayer='X'
        self.minPlayer='O'
        self.maxDepth=3
        #The start indexes of each local board
        self.globalIdx=[(0,0),(0,3),(0,6),(3,0),(3,3),(3,6),(6,0),(6,3),(6,6)]

        #Start local board index for reflex agent playing
        self.startBoardIdx=4
        #utility value for reflex offensive and reflex defensive agents
        self.winnerMaxUtility=10000
        self.twoInARowMaxUtility=500
        self.preventThreeInARowMaxUtility=100
        self.cornerMaxUtility=30

        self.winnerMinUtility=-10000
        self.twoInARowMinUtility=-100
        self.preventThreeInARowMinUtility=-500
        self.cornerMinUtility=-30

        self.expandedNodes=0
        self.currPlayer=True
        self.blockDict = {}
        self.INF = 100000000
        self.steps = 0
        for blockIndex in range(9):
            self.blockDict[blockIndex] = True
    def setStart(self, start):
        self.startBoardIdx=start
    def randomStart(self):
        self.startBoardIdx = randint(0,8)
        # print(self.startBoardIdx)
    def reInit(self):
        for blockIndex in range(9):
            self.blockDict[blockIndex] = True   
        for y in range(9):
            for x in range(9):
                self.board[y][x] = '_'     
    def printGameBoard(self):
        """
        This function prints the current game board.
        """
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[:3]])+'\n')
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[3:6]])+'\n')
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[6:9]])+'\n')


    def evaluatePredifined(self, isMax):
        """
        This function implements the evaluation function for ultimate tic tac toe for predifined agent.
        input args:
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        score(float): estimated utility score for maxPlayer or minPlayer
        """
        #YOUR CODE HERE
        score=0
        if isMax:
            if self.checkWinner()==1:
                return 10000
            score += self.checkTwo(isMax)
            if score > 0:
                return score
            score += self.checkAllCorners(True)
        else:
            if self.checkWinner()==-1:
                return -10000
            score -= self.checkTwo(isMax)
            if score < 0:
                return score
            score -= self.checkAllCorners(False)
        return score

    def checkTwo(self,isMax):
        score = 0 
        for blockIndex in range(9):
            score += self.checkLocalTwo(blockIndex,isMax)
        return score

    def checkLocalTwo(self, blockIndex, isMax):
        score = 0
        blockCount = [0]
        twoRowCount = [0]
        blockStart = self.globalIdx[blockIndex]
        target = 'x' if isMax else 'o'
        component = 'o' if isMax else 'x'
        for i in range(3):
            self.checkRow([blockStart[0]+i,blockStart[1]],target,target,'_',twoRowCount)
            self.checkRow([blockStart[0]+i,blockStart[1]],'_',target,target,twoRowCount)
            self.checkRow([blockStart[0]+i,blockStart[1]],target,'_',target,twoRowCount)
            self.checkRow([blockStart[0]+i,blockStart[1]],component,component,target,blockCount)
            self.checkRow([blockStart[0]+i,blockStart[1]],target,component,component,blockCount)
            self.checkRow([blockStart[0]+i,blockStart[1]],component,target,component,blockCount)

            self.checkCol([blockStart[0],blockStart[1]+i],target,target,'_',twoRowCount)        
            self.checkCol([blockStart[0],blockStart[1]+i],'_',target,target,twoRowCount)     
            self.checkCol([blockStart[0],blockStart[1]+i],target,'_',target,twoRowCount)        
            self.checkCol([blockStart[0],blockStart[1]+i],component,component, target,blockCount)
            self.checkCol([blockStart[0],blockStart[1]+i],target,component,component,blockCount)        
            self.checkCol([blockStart[0],blockStart[1]+i],component,target,component,blockCount)       

        self.checkLeftUp(blockStart,target,target,'_',twoRowCount)
        self.checkLeftUp(blockStart,'_',target,target,twoRowCount) 
        self.checkLeftUp(blockStart,target,'_',target,twoRowCount) 
        self.checkLeftUp(blockStart,target,component,component,blockCount) 
        self.checkLeftUp(blockStart,component,target,component,blockCount) 
        self.checkLeftUp(blockStart,component,component,target,blockCount) 

        self.checkRightUp(blockStart,target,target,'_',twoRowCount)
        self.checkRightUp(blockStart,'_',target,target,twoRowCount) 
        self.checkRightUp(blockStart,target,'_',target,twoRowCount) 
        self.checkRightUp(blockStart,target,component,component,blockCount) 
        self.checkRightUp(blockStart,component,target,component,blockCount) 
        self.checkRightUp(blockStart,component,component,target,blockCount) 

        if isMax:
            score += twoRowCount[0] * 500 + blockCount[0] * 100
        else:
            score += twoRowCount[0] * 100 + blockCount[0] * 500
        return score 


    def checkLeftUp(self, start, obj1, obj2, obj3, count):
        if self.board[start[0]][start[1]] == obj1 and self.board[start[0]+1][start[1]+1] == obj2 and self.board[start[0]+2][start[1]+2] == obj3:
            count[0] += 1
        return None

    def checkRightUp(self, start, obj1, obj2, obj3, count):
        if self.board[start[0]][start[1]] == obj1 and self.board[start[0]+1][start[1]-1] == obj2 and self.board[start[0]+2][start[1]-2] == obj3:
            count[0] += 1
        return None
    
    def checkRow(self,start, obj1, obj2, obj3, count):
        if self.board[start[0]][start[1]] == obj1 and self.board[start[0]][start[1]+1] == obj2 and self.board[start[0]][start[1]+2] == obj3:
            count[0] += 1
            return True
        return False
    
    def checkCol(self,start, obj1, obj2, obj3, count):
        if self.board[start[0]][start[1]] == obj1 and self.board[start[0]+1][start[1]] == obj2 and self.board[start[0]+2][start[1]] == obj3:
            count[0] += 1
        return None
    
    def checkAllCorners(self,isMax):
        score = 0
        for blockIndex in range(9):
            score += self.checkLocalCorner(blockIndex,isMax)
        return score

    def checkLocalCorner(self,blockIndex,isMax):
        score = 0
        block = self.globalIdx[blockIndex]
        for x in range(2):
            for y in range(2):
                if isMax:
                    if self.board[block[0]+y*2][block[1]+x*2] == 'x':
                        score += 30
                else:
                    if self.board[block[0]+y*2][block[1]+x*2] == 'o':
                        score += 30
        return score
    def twoInRow(self,blockIndex):
        res = False
        score = 0
        twoRowCount = [0]
        blockStart = self.globalIdx[blockIndex]
        target = 'x' 
        component = 'o'
        for i in range(3):
            if self.checkRow([blockStart[0]+i,blockStart[1]],target,target,'_',twoRowCount) or\
            self.checkRow([blockStart[0]+i,blockStart[1]],'_',target,target,twoRowCount) or \
            self.checkRow([blockStart[0]+i,blockStart[1]],target,'_',target,twoRowCount) or \
            self.checkCol([blockStart[0],blockStart[1]+i],target,target,'_',twoRowCount) or \
            self.checkCol([blockStart[0],blockStart[1]+i],'_',target,target,twoRowCount) or \
            self.checkCol([blockStart[0],blockStart[1]+i],target,'_',target,twoRowCount):
                return True  
        if self.checkLeftUp(blockStart,target,target,'_',twoRowCount) or\
        self.checkLeftUp(blockStart,'_',target,target,twoRowCount) or\
        self.checkLeftUp(blockStart,target,'_',target,twoRowCount) or\
        self.checkRightUp(blockStart,target,target,'_',twoRowCount) or\
        self.checkRightUp(blockStart,'_',target,target,twoRowCount) or\
        self.checkRightUp(blockStart,target,'_',target,twoRowCount):
            return True 
        return False

    def oppoWillWin(self,move):
        res = 0
        nextBlock = self.getNextBlock(move)
        if self.twoInRow(nextBlock):
            return 1
        else:
            return 0
    def evaluateDesigned(self,isMax,move):
        """
        This function implements the evaluation function for ultimate tic tac toe for your own agent.
        input args:
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        score(float): estimated utility score for maxPlayer or minPlayer
        """
        score=0
        if isMax:
            print("off using def strat, stop!")
            exit(1)

        else:
            
            if self.checkWinner()==-1:
                return -10000
            # if self.oppoWillWin(move) == 1:
            if self.checkWinner() == 1:
                print("oppo will win")
                print("total steps",self.steps)
                return 10000
            score -= self.checkTwo(isMax)
            if score < 0:
                return score
            score -= self.checkAllCorners(False)
        return score

    def checkMovesLeft(self):
        """
        This function checks whether any legal move remains on the board.
        output:
        movesLeft(bool): boolean variable indicates whether any legal move remains
                        on the board.
        """
        potentialBlocks = [k for k, v in self.blockDict.items() if v]
        movesLeft= False
        for blockIndex in potentialBlocks:
            if self.checkLocalMoveLeft(blockIndex):
                movesLeft = True
                break
        return movesLeft
    
    def checkLocalMoveLeft(self,blockIndex):
        MoveLeft = False
        blockStart = self.globalIdx[blockIndex]
        for y in range(3):
            for x in range(3):
                if self.board[blockStart[0]+y][blockStart[1]+x] == '_':
                    MoveLeft = True
                    break
        return MoveLeft

    def checkWinner(self):
        #Return termimnal node status for maximizer player 1-win,0-tie,-1-lose
        """
        This function checks whether there is a winner on the board.
        output:
        winner(int): Return 0 if there is no winner.
                     Return 1 if maxPlayer is the winner.
                     Return -1 if miniPlayer is the winner.
        """
        countWin = [0,0]
        for blockIndex in range(9):
            self.checkBlockWinner(blockIndex,countWin)

        if countWin[0] >= 1:
            return 1
        elif countWin[1] >= 1:
            return -1
        else:
            return 0

    def checkBlockWinner(self,blockIndex,countWin):
        blockStart = self.globalIdx[blockIndex]
        startY = blockStart[0]
        startX = blockStart[1]
        for y in range(3):
            if self.board[startY+y][startX] == self.board[startY+y][startX+1] == self.board[startY+y][startX+2]:
                if self.board[startY+y][startX] == 'x':
                    countWin[0] += 1
                elif self.board[startY+y][startX] == 'o':
                    countWin[1] += 1
                else: 
                    pass
        for x in range(3):
            if self.board[startY][startX+x] == self.board[startY+1][startX+x] == self.board[startY+2][startX+x]:
                if self.board[startY][startX+x] == 'x':
                    countWin[0] += 1
                elif self.board[startY][startX+x] == 'o':
                    countWin[1] += 1
                else: 
                    pass
        if self.leftUpEqual(startX,startY) or self.rightUpEqual(startX+2,startY):
            if self.board[startY+1][startX+1] == 'x':
                countWin[0] += 1

            elif self.board[startY+1][startX+1] == 'o':
                countWin[1] += 1

            else: 
                pass        
        
        return None


    def alphabeta(self, depth, blockIndex,isMax, move, alpha, beta,expandedNode, userEval):
        """
        This function implements alpha-beta algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        expandedNode[0] += 1
        if depth >= self.maxDepth-1 or not self.checkMovesLeft():
            # if isMax: # if new step is def
            #     return self.evaluateDesigned(isMax,move), move
            # else:
            return self.evaluatePredifined(isMax), move
        availableLocalSpots = self.getAvailableLocalSpots(blockIndex)
        bestValue = -1000000 if isMax else 1000000
        bestMove = []
        if availableLocalSpots: # local 
            for spot in availableLocalSpots:
                self.draw(spot,isMax)
                tempBlockIndex = spot[0]%3*3 + spot[1]%3;
                # one problem is that whether we should predict the next step using alphabeta?
                tempValue,tempMove = self.alphabeta(depth+1,tempBlockIndex,not isMax, spot, alpha, beta,expandedNode, userEval)
                bestValue = self.getBetterValue(tempValue,bestValue,isMax)
                bestMove = self.getBetterMove(tempValue,bestValue,isMax,spot,bestMove)
                self.erase(spot)
                if isMax:
                    alpha[0] = max(alpha[0],bestValue)
                    if beta[0] <= alpha[0]:
                        break
                else:
                    beta[0] = min(beta[0],bestValue)
                    if beta[0] <= alpha[0]:
                        break
            return bestValue, bestMove
        else: # global
            availableGlobalSpots = self.getAvailableGlobalSpots(blockIndex)
            for spot in availableGlobalSpots:
                self.draw(spot,isMax)
                tempBlockIndex = spot[0]%3*3 + spot[1]%3;
                tempValue,tempMove = self.alphabeta(depth+1,tempBlockIndex,not isMax, spot, alpha,beta,expandedNode, userEval)
                bestValue = self.getBetterValue(tempValue,bestValue,isMax)
                bestMove = self.getBetterMove(tempValue,bestValue,isMax,spot,bestMove)
                self.erase(spot)
                if isMax:
                    alpha[0] = max(alpha[0],bestValue)
                    if beta[0] <= alpha[0]:
                        break
                else:
                    beta[0] = min(beta[0],bestValue)
                    if beta[0] <= alpha[0]:
                        break

            return bestValue, bestMove

        # return bestValue,bestMove

    def getBlock(self,cur): # get the index of block where cur is 
        row = cur[0]//3
        col = cur[1]//3
        return (int)(col + row * 3)
    
    def updateWin(self, move, curWin):
        blockIndex = self.getBlock(move)
        blockStart = self.globalIdx[blockIndex]
        startY = blockStart[0]
        startX = blockStart[1]
        for y in range(3):
            if self.board[startY+y][startX] == self.board[startY+y][startX+1] == self.board[startY+y][startX+2]:
                if self.board[startY+y][startX] == 'x':
                    curWin[0] += 1
                    self.blockDict[blockIndex] = False
                    return None
                elif self.board[startY+y][startX] == 'o':
                    curWin[1] += 1
                    self.blockDict[blockIndex] = False
                    return None
                else: 
                    pass
        for x in range(3):
            if self.board[startY][startX+x] == self.board[startY+1][startX+x] == self.board[startY+2][startX+x]:
                if self.board[startY][startX+x] == 'x':
                    curWin[0] += 1
                    self.blockDict[blockIndex] = False
                    return None
                elif self.board[startY][startX+x] == 'o':
                    curWin[1] += 1
                    self.blockDict[blockIndex] = False
                    return None
                else: 
                    pass
        if self.leftUpEqual(startX,startY) or self.rightUpEqual(startX+2,startY):
            if self.board[startY+1][startX+1] == 'x':
                curWin[0] += 1
                self.blockDict[blockIndex] = False
                return None

            elif self.board[startY+1][startX+1] == 'o':
                curWin[1] += 1
                self.blockDict[blockIndex] = False
                return None
            else: 
                pass

    def leftUpEqual(self,startX,startY):
        return self.board[startY][startX] == self.board[startY+1][startX+1] == self.board[startY+2][startX+2]

    def rightUpEqual(self,startX,startY):
        return self.board[startY][startX] == self.board[startY+1][startX-1] == self.board[startY+2][startX-2]

    def playGamePredifinedAgent(self,maxFirst,isMinimaxOffensive,isMinimaxDefensive):
        ending = False
        offense = maxFirst
        bestMoves = []
        bestValues = []
        curBestValue = 0.0
        curMove = []
        winner = 0
        curWin = [0,0]
        expandedNodeOffense = [0] 
        expandedNodeDefence = [0]
        blockIndex = self.startBoardIdx
        while not ending:
            tempAlpha =[-self.INF]
            tempBeta = [self.INF]
            if offense:
                if isMinimaxOffensive:
                    curBestValue, curMove = self.minimax(0,blockIndex,True,[],expandedNodeOffense)
                else:
                    curBestValue, curMove = self.alphabeta(0,blockIndex,True,[],tempAlpha,tempBeta,expandedNodeOffense,False)
            else:
                if isMinimaxDefensive:
                    curBestValue, curMove = self.minimax(0,blockIndex,False,[],expandedNodeDefence)
                else:
                    curBestValue, curMove = self.alphabeta(0,blockIndex,False,[],tempAlpha,tempBeta,expandedNodeDefence,False)
            blockIndex = self.getNextBlock(curMove)
            bestMoves.append(curMove)
            bestValues.append(curBestValue)
            self.board[curMove[0]][curMove[1]] = 'x' if offense else 'o'
            offense = not offense
            self.updateWin(curMove,curWin)
            if curWin[0] == 1 or curWin[1] == 1 or not self.checkMovesLeft(): # win 3 blocks
                ending = True
        if curWin[0]>curWin[1]:
            winner = 1
        elif curWin[1] > curWin[0]:
            winner = -1
        else:
            winner = 0
        # print(curWin)
        print("offense expanded nodes is",expandedNodeOffense)
        print("defense expanded nodes is",expandedNodeDefence)

        return bestMoves, bestValues, winner
        
    def getNextBlock(self,curMove):
        return curMove[0]%3*3 + curMove[1]%3

    def minimax(self, depth, blockIndex,isMax, move, expandedNode):
        expandedNode[0] += 1
        if depth >= self.maxDepth-1 or not self.checkMovesLeft():
            return self.evaluatePredifined(isMax), move
        availableLocalSpots = self.getAvailableLocalSpots(blockIndex)
        bestValue = -1000000 if isMax else 1000000
        bestMove = []
        if availableLocalSpots: # local 
            for spot in availableLocalSpots:
                self.draw(spot,isMax)
                tempBlockIndex = spot[0]%3*3 + spot[1]%3;
                # one problem is that whether we should predict the next step using minimax?
                tempValue,tempMove = self.minimax(depth+1,tempBlockIndex,not isMax, spot,expandedNode)
                bestValue = self.getBetterValue(tempValue,bestValue,isMax)
                bestMove = self.getBetterMove(tempValue,bestValue,isMax,spot,bestMove)
                self.erase(spot)
        else: # global
            availableGlobalSpots = self.getAvailableGlobalSpots(blockIndex)
            for spot in availableGlobalSpots:
                self.draw(spot,isMax)
                tempBlockIndex = spot[0]%3*3 + spot[1]%3;
                tempValue,tempMove = self.minimax(depth+1,tempBlockIndex,not isMax, spot,expandedNode)
                bestValue = self.getBetterValue(tempValue,bestValue,isMax)
                bestMove = self.getBetterMove(tempValue,bestValue,isMax,spot,bestMove)
                self.erase(spot)
        return bestValue,bestMove

    def getBetterValue(self,tempValue,bestValue,isMax):
        return max(tempValue,bestValue) if isMax else min(tempValue,bestValue)

    def getBetterMove(self,tempValue,bestValue,isMax,tempMove,bestMove):
        if not bestMove:
            return tempMove
        elif self.getBetterValue(tempValue,bestValue,isMax) == bestValue:
            return bestMove
        else:
            return tempMove

    def getLocalSpots(self, blockIndex):
        res = []
        startPos = self.globalIdx[blockIndex]
        for i in range(3):
            for j in range(3):
                res.append([startPos[0]+i,startPos[1]+j]) 
        return res
    def getAvailableLocalSpots(self,blockIndex):
        if self.blockDict[blockIndex] == False:
            return []
        return [spot for spot in self.getLocalSpots(blockIndex) if self.board[spot[0]][spot[1]] == '_']

    def getAvailableGlobalSpots(self,blockIndex):
        res = []
        availableBlock = []
        for i in range(9):
            if self.blockDict[i] == True:
                availableBlock.append(i)
        for blockIndex in availableBlock:
            res.extend(self.getAvailableLocalSpots(blockIndex))
        return res

    def draw(self,spot,isMax):
        content = 'x' if isMax else 'o'
        self.board[spot[0]][spot[1]] = content

    def erase(self,spot):
        self.board[spot[0]][spot[1]] = '_'

    def playGameYourAgent(self,maxFirst):
        """
        This function implements the processes of the game of your own agent vs predifined offensive agent.
        input args:
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        ending = False
        offense = maxFirst
        bestMoves = []
        bestValues = []
        curBestValue = 0.0
        curMove = []
        winner = 0
        curWin = [0,0]
        expandedNodeOffense = [0] 
        expandedNodeDefence = [0]
        blockIndex = self.startBoardIdx
        while not ending:
            self.steps+=1
            tempAlpha =[-self.INF]
            tempBeta = [self.INF]
            if offense:
                curBestValue, curMove = self.alphabeta(0,blockIndex,True,[],tempAlpha,tempBeta,expandedNodeOffense, False)
            else:
                curBestValue, curMove = self.alphabeta(0,blockIndex,False,[],tempAlpha,tempBeta,expandedNodeDefence, True)
            blockIndex = self.getNextBlock(curMove)
            bestMoves.append(curMove)
            bestValues.append(curBestValue)
            self.board[curMove[0]][curMove[1]] = 'x' if offense else 'o'
            offense = not offense
            self.updateWin(curMove,curWin)
            if curWin[0] == 1 or curWin[1] == 1 or not self.checkMovesLeft(): # win 3 blocks
                ending = True
        if curWin[0]>curWin[1]:
            winner = 1
        elif curWin[1] > curWin[0]:
            winner = -1
        else:
            winner = 0
        # print(curWin)
        # print("offense expanded nodes is",expandedNodeOffense)
        # print("defense expanded nodes is",expandedNodeDefence)
        return bestMoves, bestValues, winner


    def playGameHuman(self):
        """
        This function implements the processes of the game of your own agent vs a human.
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        #YOUR CODE HERE
        bestMove=[]
        gameBoards=[]
        winner=0
        return gameBoards, bestMove, winner

def test1():
    print("task1")
    uttt=ultimateTicTacToe()
    print("========================")
    print("off:minimax, def:minimax")
    bestMove, bestValue, winner=uttt.playGamePredifinedAgent(True,True,True)
    uttt.printGameBoard()
    if winner == 1:
        print("The winner is maxPlayer!!!")
    elif winner == -1:
        print("The winner is minPlayer!!!")
    else:
        print("Tie. No winner:(")
    print("========================")

    print("off: minimax, def: alphabeta")
    uttt.reInit()
    bestMove, bestValue, winner=uttt.playGamePredifinedAgent(True,True,False)
    uttt.printGameBoard()
    if winner == 1:
        print("The winner is maxPlayer!!!")
    elif winner == -1:
        print("The winner is minPlayer!!!")
    else:
        print("Tie. No winner:(")
    print("========================")

    print("off: alphabeta, def:minimax")
    uttt.reInit()
    bestMove, bestValue, winner=uttt.playGamePredifinedAgent(False,False,True)
    # print(bestMove)
    uttt.printGameBoard()
    if winner == 1:
        print("The winner is maxPlayer!!!")
    elif winner == -1:
        print("The winner is minPlayer!!!")
    else:
        print("Tie. No winner:(")
    print("========================")

    print("off: alphabeta, def: alphabeta")
    uttt.reInit()
    bestMove, bestValue, winner=uttt.playGamePredifinedAgent(False,False, False)
    # print(bestMove)
    uttt.printGameBoard()
    if winner == 1:
        print("The winner is maxPlayer!!!")
    elif winner == -1:
        print("The winner is minPlayer!!!")
    else:
        print("Tie. No winner:(")    

def test2():
    offWin = 0
    defWin = 0
    uttt=ultimateTicTacToe()
    for i in range(1):
        uttt.reInit()
        # uttt.randomStart()
        uttt.setStart(6)
        # start = choice([True,False])
        start = True
        bestMove, bestValue, winner=uttt.playGameYourAgent(start)
        print("Max start?", start)
        uttt.printGameBoard()
        if winner == 1:
            print("The winner is maxPlayer!!!")
            offWin += 1
        elif winner == -1:
            print("The winner is minPlayer!!!")
            defWin += 1
        else:
            print("Tie. No winner:(")      
    print("offWin is ",offWin)
    print("defWin is ",defWin)

if __name__=="__main__":
    test1()
    # print("Part2")
    # test2()




