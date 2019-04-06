from typing import Tuple, List
Interval = Tuple[float, float]
Nodes = List[float]


import numpy as np

class Partition:
    """
    Represents an interval partition given some nodes

    ...

    Attributes
    ----------
    points : List[float]
        A list of nodal points including the extrema of the interval, the interval is considered to be [a,b)
    intervals : Tuple[float, float]
        A list of tuples (x_k, x_(k+1)) representing the interval [x_k, x_(k+1) )
    N : int
        The number of intervals in the partition
    Methods
    -------
    __getitem__()
        Returns the n_th interval
    IntervalFromPoint(x : float)
        Returns the interval in which x belongs, using intervals of type [,)
    """    
    
    @staticmethod
    def Uniform(a : float, b : float, n : int) -> 'Partition':
        """
        Creates an uniform partition of [a,b) with n intervals.
        
        Parameters
        ----------
        a : float
            The left limit of the interval
        b : float
            The right limit of the interval, open
        n : int
            Number of intervals in the partition
        """        
        return Partition(np.linspace(a,b,n+1))
        
    def __getitem__(self, key : int) -> Interval:
        return self.intervals[key]

    def __init__(self, points : Nodes):
        self.points = np.asarray(points)
        self.N = len(points) - 1 #Number of intervals
        
        self.intervals = []
        for i in range(self.N):
            self.intervals.append( (self.points[i], self.points[i+1]) )
            
    def IntervalFromPoint(self, x : float) -> Interval:
        part = self.points
        left = 0
        right = self.N
        #Binary search
        while right - left > 1:
            mid = int(np.floor((left+right)/2))
            if x < part[mid]:
                right = mid
            elif x > part[mid]:
                left = mid
            else:
                left = mid
                right = mid + 1
                
        return left