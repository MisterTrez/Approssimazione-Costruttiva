from typing import Callable

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

import Approximation

def L1Norm(approx : Approximation, points : int = 25, mon = False) -> float:
    return LpNorm(approx, 1, points, mon)

def L2Norm(approx : Approximation, points : int = 25, mon = False) -> float:
    return LpNorm(approx, 2, points, mon)

def L2TrapNorm(approx : Approximation, points : int = 25, mon = False) -> float:
    errF = lambda x : np.abs(approx.target(x) - approx(x)) ** 2
    errFVec = np.vectorize(errF)
    tot = 0.0
    for i, intv in enumerate(approx.partition.intervals):
        space = np.linspace(intv[0], intv[1], points)
        err = errFVec(space)
        tot += scipy.integrate.trapz(err, space)
        
    return np.sqrt(tot)

def LpNorm(approx : Approximation, p : float, points : int = 25, mon = False) -> float:
    """
    Compute the L_p norm of the error.

    Parameters
    ----------
    approx : Approximation
        The approximation we wish to evluate
    p : The p value of the L_p space.
    points : int, optional
        The number of points to split each interval in for error sampling.
    mon : bool, opt
        Monotonicity of residual shortcut, ONLY WORKS FOR CONSTANT APPROXIMATION
    """
    
    errF = lambda x : np.abs(approx.target(x) - approx(x)) ** p
    errFVec = np.vectorize(errF)
    a = approx.partition.points[0]
    b = approx.partition.points[-1]
    return scipy.integrate.quad(errFVec, a, b)[0] ** (1 / p)
    
def MaxNorm(approx : Approximation, points : int = 25, mon = False) -> float:
    """
    Compute the L_infty norm of the error.

    Parameters
    ----------
    approx : Approximation
        The approximation we wish to evluate
    points : int, optional
        The number of points to split each interval in for error sampling.
    mon : bool, opt
        Monotonicity of residual shortcut, ONLY WORKS FOR CONSTANT APPROXIMATION
    """

    if not mon:
        errF = lambda x : np.abs(approx.target(x) - approx(x))
        errFVec = np.vectorize(errF)
        error = 0.0

        for i, intv in enumerate(approx.partition.intervals):
            space = np.linspace(intv[0],intv[1], points)
            error = max(error, np.max(errFVec(space)))

        return error

    elif mon:
        err = 0.0
        fa = None
        fb = approx.target(approx.partition.points[0])
        for i, intv in enumerate(approx.partition.intervals):
            fa = fb
            fb = approx.target(intv[1])

            el = abs(fa - approx(intv[0]) )
            er = abs(fb - approx(intv[1]) )
            locerr = max(el, er)
            err = max(err, locerr)

        return err
        
class ErrorManager:
    """
    Computes and plots EOC estimates

    ...

    Attributes
    ----------
    norm : Callable
        The norm to use to compute the error
    errors : List[float]
        The error in each run
    dofs : List[int]
        The number of dofs in each run
    EOC : List[float]
        The computed EOC at each step, nan in the first run
    Methods
    -------
    PushError(approx, points, mon)
        Adds another run to the EOC calculations
    PlotEOC(ax, ymax)
        Plots the eoc graph
    """
    
    
    def __init__(self, norm : Callable):
        """
        Builds an error managed based on a specific norm.
        
        Parameters
        ----------
        norm : function
            The error norm
        """
        
        self.norm = norm
        self.errors = []
        self.dofs = []
        self.EOC = []
        
    def PushError(self, approx  : Approximation, points : int = 25, mon = False):
        """
        Records another approximation error to update EOC estimates.
        
        Parameters
        ----------
        approx : Approximation
            The latest approximation
        points : int, opt
            The number of points on which we sample error in each interval
        mon : bool, opt
            Monotonicity of residual shortcut, ONLY WORKS FOR CONSTANT APPROXIMATION
        """
        
        nor = self.norm(approx, points, mon)
        self.errors.append(nor)
        self.dofs.append(approx.partition.N)
        
        n = len(self.errors) - 1
        if n == 0 or self.errors[n] == 0:
            self.EOC.append(np.nan)
        else:
            self.EOC.append( math.log(self.errors[n-1]/self.errors[n], 2) / math.log(self.dofs[n]/self.dofs[n-1], 2)  )
            
    def PlotEOC(self, ax, ymax : float = 1.1):
        """
        Plots the EOC graph.
        
        Parameters
        ----------
        ymax : float
            The maximum y on the graph
        """
        
        if ymax == 1.1:
            maxEOC = np.nanmax(self.EOC)
            yround = math.ceil(maxEOC)
            ymax = yround + 0.1
        
        
        ax.set_title("EOC")
        ax.set_xlabel("N")
        ax.set_ylabel("EOC_n")
        ax.set_ylim((0.0, ymax))
        for i in range(0, math.ceil(ymax)):
            ax.axhline(i, color="r", dashes=[10,10], linewidth = 0.5)
        ax.plot(self.dofs, self.EOC)