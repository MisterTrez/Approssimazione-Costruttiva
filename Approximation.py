from typing import Callable, List

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

import Partition

class Approximation:
    """
    Approximates a target function using a finite-dimensional functional space determined by a partition.

    ...

    Attributes
    ----------
    target : Callable
        The function we wish to approximate
    partition : Partition
        The partition on which the approximation occurs.
    dof : List[float]
        The coefficients needed to uniquely identify the function.
    NDOF : int
        The number of parameters needed to uniquely identify the function
    Methods
    -------
    InitializeDof()
        Called by constructor, fills the "dof" array to build an approximant. [ABSTRACT]
    Evaluate(x : float, dofs : List[float], points : List[float])
        Evaluates the function on a single interval, using the dofs and points associated with that interval. [ABSTRACT]
    __call__ (x : float)
        Assembles the relevant dof data and evaluates the approximant at the given point.
    plot(ax, points : int, opt)
        Plots the target function and its approximant by sampling it at "points" points
    """
    
    def __init__(self, partition : Partition, target : Callable):
        self.target = target
        self.partition = partition
        #self.continuous = False
        self.InitializeDof()
        
    def InitializeDof(self):
        #set dof and interval_to_dof
        pass
    
    def Evaluate(self, x : float, dofs : List[float], points : List[float]):
        pass
    
    def __call__(self, x : float):
        interval = self.partition.IntervalFromPoint(x)
        dofs = [self.dof[i] for i in self.IntToDof[interval]]
        points = [self.partition.points[i] for i in self.IntToDof[interval]]
        return self.Evaluate(x, dofs, points)
    
    
    def plot(self, ax = None, points : int = 25):
        #plt.figure(figsize=(12,6))
        targetVec = np.vectorize(self.target)
        approxVec = np.vectorize(self)
        intervals = self.partition.intervals
        
        ones = np.ones(points)
        
        hasAx = True
        
        if ax is None:
            ax = plt
            hasAx = False
        
        for i,intv in enumerate(intervals):           
            space = np.linspace(intv[0],intv[1],points)
            vals = targetVec(space)
            ax.plot(space, vals, "blue")
            
            values = approxVec(space)
            if not self.continuous:
                values[-1] = self.dof[i]
                
            ax.plot(space, values, "red")
            
        if hasAx:   
            ax.set_title("f and its approx")
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
                    
        
        
class MidpointApproximation(Approximation):
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

class L2ConstantApproximation(Approximation):
    def __init__(self, partition, target):
        self.continuous = False
        
        Approximation.__init__(self, partition, target)
        
    def InitializeDof(self):
        self.NDOF = self.partition.N
        self.dof = np.zeros(self.NDOF)
        self.IntToDof = []
        
        for i, intv in enumerate(self.partition.intervals):
            lnt = intv[1] - intv[0]
            self.dof[i] = scipy.integrate.quad(self.target, intv[0], intv[1])[0]  / lnt
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