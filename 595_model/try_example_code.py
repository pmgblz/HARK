"""
Trying out different model specifications. 
I'm not sure how to implement the model we are currently running. 
"""


# CHANGE PATH - WORKING DIRECTORY
# select your path
path = "C:/Users/Paula Gablenz/Dropbox/000_RA/GitHub/HARK/ConsumptionSaving"

import os
retval = os.getcwd()
print "Current working directory %s" % retval
# Change directory
os.chdir( path )

retval = os.getcwd()
print "Directory changed successfully %s" % retval


# IMPORT 

from TractableBufferStockModel import TractableConsumerType
from ConsIndShockModel import IndShockConsumerType
from ConsIndShockModel import PerfForesightConsumerType

import matplotlib.pyplot as plt
import numpy as np                   # numeric Python
from HARKutilities import plotFuncs  # basic plotting tools
from ConsMarkovModel import MarkovConsumerType # An alternative, much longer way to solve the TBS model
from time import clock               # timing utility

import ConsumerParameters as Params
from HARKutilities import plotFuncsDer, plotFuncs

mystr = lambda number : "{:.4f}".format(number)

do_simulation           = True

###########################################################################
# TRY TO SOLVE CONSUMER IDTIOSYNCRATIC SHOCK MODEL FROM SEPARATE FILE #
###########################################################################

# Make and solve an idiosyncratic shocks consumer with a finite lifecycle
LifecycleExample = IndShockConsumerType(**Params.init_lifecycle)
LifecycleExample.cycles = 1 # Make this consumer live a sequence of periods exactly once

start_time = clock()
LifecycleExample.solve()
end_time = clock()
print('Solving a lifecycle consumer took ' + mystr(end_time-start_time) + ' seconds.')
LifecycleExample.unpackcFunc()
LifecycleExample.timeFwd()

# Plot the consumption functions during working life
print('Consumption functions while working:')
mMin = min([LifecycleExample.solution[t].mNrmMin for t in range(LifecycleExample.T_cycle)])
plotFuncs(LifecycleExample.cFunc[:LifecycleExample.T_retire],mMin,5)

# Plot the consumption functions during retirement
print('Consumption functions while retired:')
plotFuncs(LifecycleExample.cFunc[LifecycleExample.T_retire:],0,5)
LifecycleExample.timeRev()

# Simulate some data; results stored in mNrmNow_hist, cNrmNow_hist, pLvlNow_hist, and t_age_hist
if do_simulation:
    LifecycleExample.T_sim = 120
    LifecycleExample.track_vars = ['mNrmNow','cNrmNow','pLvlNow','t_age']
    LifecycleExample.initializeSim()
    LifecycleExample.simulate()
    

###############################################################################
# CREATE OWN CONSUMER TYPE - FOLLOWING P.15 OF THE HANDBOOK  
###############################################################################
    
# Added example consumer     
    
print('Plotting Consumer Example:')
    
MyConsumer = PerfForesightConsumerType(time_flow=True, cycles=1,
CRRA = 2.7, Rfree = 1, DiscFac = 0.98,
LivPrb = [0.99,0.98,0.97,0.96,0.95,0.94,0.93,0.92,0.91,0.90],
PermGroFac = [1.01,1.01,1.01,1.01,1.01,1.02,1.02,1.02,1.02,1.02],
Nagents=1000)
MyConsumer.solve()
    
# Plot & Create Solution
nMin = MyConsumer.solution[0].mNrmMin
plotFuncs(MyConsumer.solution[0].cFunc, mMin, mMin+10)

    

# EXAMPLE CODE 

# =============================================================================
# do_simulation = True
# 
# # Define the model primitives
# base_primitives = {'UnempPrb' : .00625,    # Probability of becoming unemployed
#                    'DiscFac' : 0.996,      # Intertemporal discount factor - Source: Ganong, Noel (2017)
#                    'Rfree' : 1.01,         # Risk-free interest factor on assets
#                    'PermGroFac' : 1.0025,  # Permanent income growth factor (uncompensated)
#                    'CRRA' : 1.0}           # Coefficient of relative risk aversion
#                    
# # Define a dictionary to be used in case of simulation
# simulation_values = {'aLvlInitMean' : 0.0,  # Mean of log initial assets for new agents 
#                      'aLvlInitStd' : 1.0,   # Stdev of log initial assets for new agents
#                      'AgentCount' : 10000,  # Number of agents to simulate
#                      'T_sim' : 120,         # Number of periods to simulate
#                      'T_cycle' : 1}         # Number of periods in the cycle
#                                             
# # Make and solve a tractable consumer type
# ExampleType = TractableConsumerType(**base_primitives)
# t_start = clock()
# ExampleType.solve()
# t_end = clock()
# print('Solving a tractable consumption-savings model took ' + str(t_end-t_start) + ' seconds.')
# 
# # Plot the consumption function and whatnot
# m_upper = 1.5*ExampleType.mTarg
# conFunc_PF = lambda m: ExampleType.h*ExampleType.PFMPC + ExampleType.PFMPC*m
# plotFuncs([ExampleType.solution[0].cFunc,ExampleType.mSSfunc,ExampleType.cSSfunc],0,m_upper)
# plotFuncs([ExampleType.solution[0].cFunc,ExampleType.solution[0].cFunc_U],0,m_upper)
# plotFuncs([ExampleType.solution[0].cFunc],0,m_upper)
# 
# if do_simulation:
#     ExampleType(**simulation_values) # Set attributes needed for simulation
#     ExampleType.track_vars = ['mLvlNow']
#     ExampleType.makeShockHistory()
#     ExampleType.initializeSim()
#     ExampleType.simulate()
#     
# 
# # Now solve the same model using backward induction rather than the analytic method of TBS.
# # The TBS model is equivalent to a Markov model with two states, one of them absorbing (permanent unemployment).
# MrkvArray = np.array([[1.0-base_primitives['UnempPrb'],base_primitives['UnempPrb']],[0.0,1.0]]) # Define the two state, absorbing unemployment Markov array
# init_consumer_objects = {"CRRA":base_primitives['CRRA'],
#                         "Rfree":np.array(2*[base_primitives['Rfree']]), # Interest factor (same in both states)
#                         "PermGroFac":[np.array(2*[base_primitives['PermGroFac']/(1.0-base_primitives['UnempPrb'])])], # Unemployment-compensated permanent growth factor
#                         "BoroCnstArt":None,   # Artificial borrowing constraint
#                         "PermShkStd":[0.0],   # Permanent shock standard deviation
#                         "PermShkCount":1,     # Number of shocks in discrete permanent shock distribution
#                         "TranShkStd":[0.0],   # Transitory shock standard deviation
#                         "TranShkCount":1,     # Number of shocks in discrete permanent shock distribution
#                         "T_cycle":1,          # Number of periods in cycle
#                         "UnempPrb":0.0,       # Unemployment probability (not used, as the unemployment here is *permanent*, not transitory)
#                         "UnempPrbRet":0.0,    # Unemployment probability when retired (irrelevant here)
#                         "T_retire":0,         # Age at retirement (turned off)
#                         "IncUnemp":0.0,       # Income when unemployed (irrelevant)
#                         "IncUnempRet":0.0,    # Income when unemployed and retired (irrelevant)
#                         "aXtraMin":0.001,     # Minimum value of assets above minimum in grid
#                         "aXtraMax":ExampleType.mUpperBnd, # Maximum value of assets above minimum in grid
#                         "aXtraCount":48,      # Number of points in assets grid
#                         "aXtraExtra":[None],  # Additional points to include in assets grid
#                         "aXtraNestFac":3,     # Degree of exponential nesting when constructing assets grid
#                         "LivPrb":[np.array([1.0,1.0])], # Survival probability
#                         "DiscFac":base_primitives['DiscFac'], # Intertemporal discount factor
#                         'AgentCount':1,       # Number of agents in a simulation (irrelevant)
#                         'tax_rate':0.0,       # Tax rate on labor income (irrelevant)
#                         'vFuncBool':False,    # Whether to calculate the value function
#                         'CubicBool':True,     # Whether to use cubic splines (False --> linear splines)
#                         'MrkvArray':[MrkvArray] # State transition probabilities
#                         }
# MarkovType = MarkovConsumerType(**init_consumer_objects)   # Make a basic consumer type
# employed_income_dist = [np.ones(1),np.ones(1),np.ones(1)]    # Income distribution when employed
# unemployed_income_dist = [np.ones(1),np.ones(1),np.zeros(1)] # Income distribution when permanently unemployed
# MarkovType.IncomeDstn = [[employed_income_dist,unemployed_income_dist]]  # set the income distribution in each state
# MarkovType.cycles = 0
# 
# # Solve the "Markov TBS" model
# t_start = clock()
# MarkovType.solve()
# t_end = clock()
# MarkovType.unpackcFunc()
# 
# print('Solving the same model "the long way" took ' + str(t_end-t_start) + ' seconds.')
# #plotFuncs([ExampleType.solution[0].cFunc,ExampleType.solution[0].cFunc_U],0,m_upper)
# plotFuncs(MarkovType.cFunc[0],0,m_upper)
# diffFunc = lambda m : ExampleType.solution[0].cFunc(m) - MarkovType.cFunc[0][0](m)
# print('Difference between the (employed) consumption functions:')
# plotFuncs(diffFunc,0,m_upper)
# 
# =============================================================================


