import pandas as PD
import numpy as np
import matplotlib.pyplot as plt

from Partition import Partition
from Approximation import MidpointApproximation, LinearContApproximation, L2ConstantApproximation
from ErrorMetrics import ErrorManager, MaxNorm, L2Norm, L2TrapNorm

def TestEOC(p, n_start, runs, a, b, appr): 
    dic = {
        "Midpoint" : MidpointApproximation, 
        "Linear" : LinearContApproximation,
        "L2" : L2ConstantApproximation
    }
    
    f = None
    
    # Define the target function
    if p > 0:
        f = lambda x : x**p
    else:
        f = lambda x : -1/(np.log(x/np.e))
    
    ns = [n_start * (2**n) for n in range(runs)] # Build the array of all the tested "n" values
    
    EM = None # Initialize the error manager with the relevant norm
    if appr == "L2":
        EM = ErrorManager(L2Norm)
    else:
        EM = ErrorManager(MaxNorm)
    
    interp = None
    for n in ns: #Foreach prescribed n
        u = Partition.Uniform(a, b, n)
        interp = dic[appr](u, f) #Build the prescribed approximation on the interval
        EM.PushError(interp, points = 50) #Give the result to the ErrorManager, so he can compute errors and EOC
        
    # Put the error data in a nice table so we can read it    
    data = {"N" : EM.dofs, "Error" : EM.errors, "EOC": EM.EOC}
    d = PD.DataFrame(data) 
    
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(16,6))
    # Set axis limits
    ax1.set_xlim((-0.05, 1.05))
    ax1.set_ylim((-0.05, 1.05))
    
    interp.plot(ax1) # Plot the function
    

    EM.PlotEOC(ax2) # Plot the EOC graph
    return d # Return the table