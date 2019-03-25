import math
import numpy as np
import matplotlib.pyplot as plt

def MaxNorm(approx, mon = False, points = 25, ):
        if not mon:
            targetVec = np.vectorize(approx.target)
            intervals = approx.partition.intervals
            highest = -1
            for i, intv in enumerate(intervals):
                space = np.linspace(intv[0],intv[1],points)
                err = np.max(np.abs(targetVec(space) - approx.dof[i]))
                if err > highest:
                    highest = err
                
            return err
        
        elif mon:
            err = 0.0
            for i, intv in enumerate(approx.partition.intervals):
                fa = approx.target(intv[0])
                fb = approx.target(intv[1])
                #print("fa " + str(fa))
                #print("fb " + str(fb))
                err = max(err, abs(fa-approx.dof[i]), abs(fb-approx.dof[i]))
                
            return err
        
class ErrorManager:
    def __init__(self, norm):
        self.norm = norm
        self.errors = []
        self.dofs = []
        self.EOC = []
        
    def PushError(self, approx, mon = False):
        self.errors.append(self.norm(approx, mon))
        self.dofs.append(approx.NDOF)
        
        n = len(self.errors) - 1
        if n == 0:
            self.EOC.append(np.nan)
        else:
            self.EOC.append( math.log(self.errors[n-1]/self.errors[n], 2) / math.log(self.dofs[n]/self.dofs[n-1], 2)  )
            
    def Plot(self):
        plt.figure(figsize=(12,6))
        plt.ylim((0.0, 1.1))
        plt.axhline(1.0, color="r", dashes=[10,10], linewidth = 0.5)
        plt.plot(self.dofs, self.EOC)
        plt.show()