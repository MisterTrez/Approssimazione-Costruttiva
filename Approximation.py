import numpy as np
import matplotlib.pyplot as plt

class Approximation:
    def __init__(self, partition, target):
        self.target = target
        self.partition = partition
        #self.continuous = False
        self.InitializeDof()
        
    def InitializeDof(self):
        #set dof and interval_to_dof
        pass
    
    def Evaluate(self, x, dofs, points):
        pass
    
    def __call__(self, x):
        interval = self.partition.intervalFromPoint(x)
        dofs = [self.dof[i] for i in self.IntToDof[interval]]
        points = [self.partition.points[i] for i in self.IntToDof[interval]]
        return self.Evaluate(x, dofs, points)
    
    
    def plot(self, points = 25):
        #plt.figure(figsize=(12,6))
        targetVec = np.vectorize(self.target)
        approxVec = np.vectorize(self)
        intervals = self.partition.intervals
        
        ones = np.ones(points)
        
        for i,intv in enumerate(intervals):           
            space = np.linspace(intv[0],intv[1],points)
            vals = targetVec(space)
            plt.plot(space, vals, "blue")
            
            values = approxVec(space)
            if not self.continuous:
                values[-1] = self.dof[i]
                
            plt.plot(space, values, "red")
              
        plt.show()        
        
        
class ConstantApproximation(Approximation):
    def __init__(self, partition, target, beta = 0.5):
        self.beta = beta
        self.continuous = False
        
        Approximation.__init__(self, partition, target)
        
    def InitializeDof(self):
        self.NDOF = self.partition.N
        self.dof = np.zeros(self.NDOF)
        self.IntToDof = []
        
        for i, intv in enumerate(self.partition.intervals):
            x_i = intv[0] + self.beta * (intv[1] - intv[0])
            self.dof[i] = self.target(x_i)
            self.IntToDof.append([i])
            
    def Evaluate(self, x, dofs, points):
        return dofs[0]
    
class LinearContApproximation(Approximation):
    def __init__(self, partition, target):
        self.continuous = True
        Approximation.__init__(self, partition, target)
        
    def InitializeDof(self):
        self.NDOF = self.partition.N + 1
        self.dof = np.zeros(self.NDOF)
        self.IntToDof = []
        
        for i, ti in enumerate(self.partition.points):
            self.dof[i] = self.target(ti)
            if i < self.NDOF:
                self.IntToDof.append([i, i+1])
            
    def Evaluate(self, x, dofs, points):
        length = points[1] - points[0]
        return dofs[0] + (dofs[1] - dofs[0]) * (x - points[0]) / length