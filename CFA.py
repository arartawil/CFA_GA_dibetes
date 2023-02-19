

import random
from CCells import CCells
from Evo import logistic,rd_2
import math
import numpy


class CFA(object):
    
    cells = [] #list of cells   

    
    SUCCESS=False
    NOFE=0 
    lastIteration=0
    
    def __init__(self, p, d,ul,ll,gp,sc,itera,r1,r2,v1,v2): 
        self.Population_Size=p
        self.Dimension=d
        self.upperLimit=ul
        self.lowerLimit=ll
        self.GlobOptimization=gp
        self.stop_Condition=sc
        self.No_Of_iteration=itera
        self.R_1=r1
        self.R_2=r2
        self.V_1=v1
        self.V_2=v2
        self.con=[0]*itera
        
    def run(self):
        Best_Cells = CCells();
        Best_Cells.Points = [0]*self.Dimension
        Best_Cells.Fitness = 100
        
        for i in range(self.Population_Size):
            self.cells.append(CCells())
                    
        for i in range(self.Population_Size):

            self.cells[i]=CCells()
            self.cells[i].Points=[] 
            for j in range(self.Dimension):
                _rand=random.gauss(100, 50)
                _rand=_rand%self.upperLimit
                #self.cells[i].Points.append(random.randint(self.lowerLimit, self.upperLimit))
                self.cells[i].Points.append(_rand)
                
            self.cells[i].Fitness=logistic(self.cells[i].Points);
            if self.cells[i].Fitness <  Best_Cells.Fitness:
                Best_Cells.Fitness=self.cells[i].Fitness
                Best_Cells.Points=self.cells[i].Points
        
        size=int(self.Population_Size/4)
        firstGroup=[CCells]* size
        secondGroup =[CCells]* size
        therdGroup =[CCells]* size
        f_g=(self.Population_Size - size * 3) 
        forthGroup=[CCells]* f_g
        L=0
        
        for i in range(size):
            firstGroup[i]=CCells()
            firstGroup[i]=self.cells[L]
            L=L+1
            
        for i in range(size):
            secondGroup[i]=CCells()
            secondGroup[i]=self.cells[L]
            L=L+1
            
        for i in range(size):
            therdGroup[i]=CCells()
            therdGroup[i]=self.cells[L]
            L=L+1            
        
        for i in range(self.Population_Size - size * 3):
            forthGroup[i]=CCells()
            forthGroup[i]=self.cells[L]
            L=L+1  
        
        f=0
        self.SUCCESS=False
        self.NOFE=0
        
        for itr in range(self.No_Of_iteration):
            #result=abs(Best_Cells.Fitness - self.GlobOptimization);
            if Best_Cells.Fitness >= self.GlobOptimization:
                result=Best_Cells.Fitness - self.GlobOptimization
            else:
                result=self.GlobOptimization-Best_Cells.Fitness
                
            if result <= self.stop_Condition:
                print("Iteration: " + str(itr))
                print([element for element in Best_Cells.Points])
                print(Best_Cells.Fitness)
                self.SUCCESS=True
                break
            
            tempPoint=[0]* self.Dimension
            refliction = 0
            visibility = 0
            av = 0
            
            for  k in range(self.Dimension):
                av += Best_Cells.Points[k];
                
            av = av / self.Dimension;
            

            for  i  in   range(len(firstGroup)):
                for j in range(self.Dimension):

                    self.R_1=random.gauss(100, 5)
                    self.R_2 = random.gauss(100, 5)
                    refliction=random.uniform(self.R_1, self.R_2) * firstGroup[i].Points[j]
                    visibility=(Best_Cells.Points[j] - firstGroup[i].Points[j])
                    
                    tempPoint[j]=refliction + visibility
                    
                    if  tempPoint[j] > self.upperLimit: tempPoint[j] = self.upperLimit;
                    if  tempPoint[j] < self.lowerLimit: tempPoint[j] = self.lowerLimit;
                    
                f = logistic(tempPoint);
                self.NOFE=self.NOFE+1;
                if f < Best_Cells.Fitness: 
                      Best_Cells.Fitness = f
                      Best_Cells.Points= tempPoint
                        

                if f < firstGroup[i].Fitness:
                    firstGroup[i].Fitness = f
                    firstGroup[i].Points=tempPoint
                
            for  i  in   range(len(secondGroup)):

                for j in range(self.Dimension):
                    
                    refliction=Best_Cells.Points[j]
                    self.V_1=random.gauss(100, 5)
                    self.V_2 = random.gauss(100, 5)
                    
                    visibility=random.uniform(self.V_1, self.V_2)* (Best_Cells.Points[j] - secondGroup[i].Points[j])
                    
                    tempPoint[j]=refliction + visibility
                    
                    if  tempPoint[j] > self.upperLimit: tempPoint[j] = self.upperLimit
                    if  tempPoint[j] < self.lowerLimit: tempPoint[j] = self.lowerLimit
                    
                f = logistic(tempPoint);
                self.NOFE=self.NOFE+1;
                if f < Best_Cells.Fitness: 
                      Best_Cells.Fitness = f;
                      Best_Cells.Points=tempPoint
                        

                if f < secondGroup[i].Fitness:
                    secondGroup[i].Fitness = f;
                    secondGroup[i].Points =tempPoint
            
            for  i  in   range(len(therdGroup)):

                _lev = self.Levy(20)
                for j in range(self.Dimension):
                    self.V_1=random.gauss(100, 5)
                    self.V_2 = random.gauss(100, 5)

                    refliction=Best_Cells.Points[j]
                    visibility=random.uniform(self.V_1, self.V_2)* (Best_Cells.Points[j] - av)
                    
                    tempPoint[j]=refliction + visibility
                    
                    if  tempPoint[j] > self.upperLimit: tempPoint[j] = self.upperLimit;
                    if  tempPoint[j] < self.lowerLimit: tempPoint[j] = self.lowerLimit;
                    
                f = logistic(tempPoint);
                self.NOFE=self.NOFE+1;
                if f < Best_Cells.Fitness: 
                      Best_Cells.Fitness = f;
                      Best_Cells.Points=tempPoint
                        

                if f < therdGroup[i].Fitness:
                    therdGroup[i].Fitness = f;
                    therdGroup[i].Points  =tempPoint
    
            for  i  in   range(len(forthGroup)):

                _lev = self.Levy(20)
                for j in range(self.Dimension):

                    refliction=random.uniform(self.lowerLimit, self.upperLimit)
                    
                    visibility=0
                    
                    tempPoint[j]=refliction + visibility
                    
                    if  tempPoint[j] > self.upperLimit: tempPoint[j] = self.upperLimit;
                    if  tempPoint[j] < self.lowerLimit: tempPoint[j] = self.lowerLimit;
                    
                f = logistic(tempPoint);
                self.NOFE=self.NOFE+1;
                if f < Best_Cells.Fitness: 
                      Best_Cells.Fitness = f;
                      Best_Cells.Points=tempPoint
                        

                if f < forthGroup[i].Fitness:
                    forthGroup[i].Fitness = f;
                    forthGroup[i].Points=tempPoint



            self.lastIteration=itr
            self.con[itr]=Best_Cells.Fitness
            #print("Iteration: "+str(itr) )
            #print( [element for element in Best_Cells.Points])
        #print(self.con)
        return Best_Cells.Points

    def Levy(self,dim):
        beta = 1.5
        sigma = (math.gamma(1 + beta)
                    * math.sin(math.pi * beta / 2)
                    / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
                ) ** (1 / beta)
        u = 0.01 * numpy.random.randn(dim) * sigma
        v = numpy.random.randn(dim)
        zz = numpy.power(numpy.absolute(v), (1 / beta))
        step = numpy.divide(u, zz)
        return step
    def calculateFitness(self,XD):
        ai = [0]*25
        bi = [0]*25
        sumation=0
        
        for  i  in range(25): 
            
            ai[i] = 16 * ((i % 5) - 2);
            bi[i] = 16 * ((i / 5) - 2);
        
            sumation = 0;
            for j in range(25):
                sumation =sumation+ (1 / (1 + j + pow(XD[0] - ai[j], 6) + pow(XD[1] - bi[j], 6)))
        
        z = 1 / 500 + sumation
        z = pow(z, -1)
        return z;
        
    




                

              
            
            
        