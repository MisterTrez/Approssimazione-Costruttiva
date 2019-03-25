import numpy as np

class Partition:
    
    @staticmethod
    def Uniform(a,b,n):
         return Partition(np.linspace(a,b,n+1))
        
    def __getitem__(self, key):
        return self.intervals[key]

    def __init__(self, points):
        self.points = np.asarray(points)
        self.N = len(points) - 1 #Number of intervals
        self.intervals = []
        for i in range(self.N):
            self.intervals.append( (self.points[i], self.points[i+1]) )