import numpy as np
import matplotlib.pyplot as plt

class ConstantApproximation:
    def __init__(self, partition, target, beta):
        self.target = target
        self.partition = partition
        self.beta = beta
        
        self.dof = np.zeros(partition.N)
        
        for i,intv in enumerate(partition.intervals):
            x_i = intv[0] + self.beta * (intv[1] - intv[0])
            self.dof[i] = self.target(x_i)
    
    def maxNormError(self, points = 25, mon = False):
        if not mon:
            targetVec = np.vectorize(self.target)
            intervals = self.partition.intervals
            highest = -1
            for i, intv in enumerate(intervals):
                space = np.linspace(intv[0],intv[1],points)
                err = np.max(np.abs(targetVec(space) - self.dof[i]))
                if err > highest:
                    highest = err
                
            return err
        
        elif mon:
            err = 0.0
            for i, intv in enumerate(self.partition.intervals):
                fa = self(intv[0])
                fb = self(intv[1])
                err = max(err, abs(fa-self.dof[i]), abs(fb-self.dof[i]))
                
            return err
    
    def __call__(self, x):
        part = self.partition.points
        left = 0
        right = self.partition.N
        while right - left > 1:
            mid = int(np.floor((left+right)/2))
            if x < part[mid]:
                right = mid
            elif x > part[mid]:
                left = mid
            else:
                left = mid
                right = mid + 1
                
        return self.dof[left]
                
    
    def plot(self, points = 25):
        targetVec = np.vectorize(self.target)
        intervals = self.partition.intervals
        
        ones = np.ones(points)
        
        for i,intv in enumerate(intervals):           
            space = np.linspace(intv[0],intv[1],points)
            vals = targetVec(space)
            plt.plot(space, vals, "blue")
            plt.plot(space, self.dof[i] * ones, "red")
              
        plt.show()
                        