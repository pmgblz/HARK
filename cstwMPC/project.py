# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 12:39:11 2017

@author: derinaksit
"""
import sys 
import os
sys.path.insert(0, os.path.abspath('../')) #Path to ConsumptionSaving folder
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../cstwMPC')) #Path to cstwMPC folder

import SetupParamsCSTW as Params
from copy import deepcopy
from scipy.optimize import golden, brentq
from time import clock
import cstwMPC
import numpy as np

#Annualize quarterly frequency parameters 
#Params.working_T=41
#Params.retired_T=55
#Params.T_sim_PY=1200
#Params.ignore_periods_PY=300
#DeprFac = 0.025**0.25

# Set targets for K/Y and the Lorenz curve based on the data
lorenz_target = cstwMPC.getLorenzShares(Params.SCF_wealth,weights=Params.SCF_weights,percentiles=Params.percentiles_to_match)
lorenz_long_data = np.hstack((np.array(0.0),cstwMPC.getLorenzShares(Params.SCF_wealth,weights=Params.SCF_weights,percentiles=np.arange(0.01,1.0,0.01).tolist()),np.array(1.0)))
KY_target = 10.26

# Construct empty arrays for MPS and aggregate saving rate
MPS=[]
ASR=[]
AtoY=[]

# Do param-dist version if True, param-point if False
#Params.do_param_dist=False

# Define the agent type
BaselineType = cstwMPC.cstwMPCagent(**Params.init_infinite)

# Exercise conducted for a range of values for PermShkStd
for k in np.arange(0.005,0.02,0.001):   
    print k
    BaselineType.PermShkStd = [(k*4/11)**0.5]
    BaselineType.TranShkStd =[(0.01*4)**0.5]
    cstwMPC.cstwMPCagent.updateIncomeProcess(BaselineType) # Crucial step for new PermShkStd value to be used 
    
    # Key parameters of the agent type     
    BaselineType.CRRA = 1.0 
    BaselineType.LivPrb = [1.0 - 1.0/160.0] 
    BaselineType.Rfree = 1.01/BaselineType.LivPrb[0]
    BaselineType.PermGroFac = [1.000**0.25]
    BaselineType.AgeDstn = np.array(1.0)
    
    # Make agent types for estimation   
    EstimationAgentList = []
    for n in range(Params.pref_type_count):
        EstimationAgentList.append(deepcopy(BaselineType))
            
    # Give all the AgentTypes different seeds
    for j in range(len(EstimationAgentList)):
        EstimationAgentList[j].seed = j
        
    # Make an economy for the consumers to live in
    EstimationEconomy = cstwMPC.cstwMPCmarket(**Params.init_market)
    EstimationEconomy.agents = EstimationAgentList
    EstimationEconomy.KYratioTarget = KY_target
    EstimationEconomy.LorenzTarget = lorenz_target
    EstimationEconomy.LorenzData = lorenz_long_data
    EstimationEconomy.PopGroFac = 1.0
    EstimationEconomy.TypeWeight = [1.0]
    EstimationEconomy.act_T = Params.T_sim_PY
    EstimationEconomy.ignore_periods = Params.ignore_periods_PY
    
    param_range = [0.982093218093,0.982093218093] #The optimal beta when cstwMPC is run with beta-dist and PermShkStd = [(0.02*4/11)**0.5]
    spread_range = [0.0122550389533,0.0122550389533] #The optimal nabla when cstwMPC is run with beta-dist and PermShkStd = [(0.02*4/11)**0.5]
        
    if Params.do_param_dist:
        # Run the param-dist estimation
        paramDistObjective = lambda spread : cstwMPC.findLorenzDistanceAtTargetKY(
                                                        Economy = EstimationEconomy,
                                                        param_name = Params.param_name,
                                                        param_count = Params.pref_type_count,
                                                        center_range = param_range,
                                                        spread = spread,
                                                        dist_type = Params.dist_type)
        t_start = clock()
        spread_estimate = golden(paramDistObjective,brack=spread_range,tol=1e-4)
        center_estimate = EstimationEconomy.center_save
        t_end = clock()
    else:
        # Run the param-point estimation only
        paramPointObjective = lambda center : cstwMPC.getKYratioDifference(Economy = EstimationEconomy,
                                             param_name = Params.param_name,
                                             param_count = Params.pref_type_count,
                                             center = center,
                                             spread = 0.0,
                                             dist_type = Params.dist_type)
        t_start = clock()
        center_estimate = brentq(paramPointObjective,param_range[0],param_range[1],xtol=1e-6)
        spread_estimate = 0.0
        t_end = clock()
            
    #Display statistics about the estimated model
    EstimationEconomy.LorenzBool = True
    EstimationEconomy.ManyStatsBool = True
    EstimationEconomy.distributeParams(Params.param_name,Params.pref_type_count,center_estimate,spread_estimate,Params.dist_type)
    EstimationEconomy.solve()
    EstimationEconomy.calcLorenzDistance()
    print('Estimate is center=' + str(center_estimate) + ', spread=' + str(spread_estimate) + ', took ' + str(t_end-t_start) + ' seconds.')
    EstimationEconomy.center_estimate = center_estimate
    EstimationEconomy.spread_estimate = spread_estimate
    EstimationEconomy.showManyStats(Params.spec_name)
    
    # Calculate and print variables of interest
    C=np.sum(np.hstack(EstimationEconomy.pLvlNow)*np.hstack(EstimationEconomy.cNrmNow)) # Aggregate Consumption Level
    A=np.sum(np.hstack(EstimationEconomy.pLvlNow)*np.hstack(EstimationEconomy.aNrmNow)) # Aggregate Assets
    M=np.sum(np.hstack(EstimationEconomy.pLvlNow)*np.hstack(EstimationEconomy.mNrmNow)) # Aggregate Market Resources
    Y=np.sum(np.hstack(EstimationEconomy.pLvlNow)*np.hstack(EstimationEconomy.TranShkNow)) # Aggregate Labor Income
    B=M-Y 
    I=(BaselineType.Rfree-1)*B+Y # Aggregate Income
    SR=(I-C)/I # Aggregate Saving Rate
    MPSall=1-EstimationEconomy.MPCall # MPS
    AY=A/Y # Asset to Income Ratio
    print(A)
    print(M)
    print(Y)
    print(C)
    print(SR)
    print(AY)
    #print(EstimationEconomy.AaggNow)
    #print(EstimationEconomy.AFunc)
    #print(np.sum(np.hstack(EstimationEconomy.aLvlNow)))

    # Construct an array for MPS and ASR to plot
    MPS.append(MPSall)
    ASR.append(SR)
    AtoY.append(AY)

#Plot the relationship between MPS/ASR and Std Dev of Perm Shk
import matplotlib.pyplot as plt
import scipy as sp
std=(np.arange(0.005,0.02,0.001)*4/11)**0.5

#plt.ylabel('MPS')
#plt.xlabel('Std Dev of Perm. Income Shock')
#plt.title('Change in Savings Following Increase in Permanent Income Uncertainty')
#plt.ylim(0.89,0.91)
#plt.scatter(std,MPS)
# Draw the linear fitted line
#m, b = np.polyfit(std, MPS, 1)
#plt.plot(std, m*std + b, '-')
#slope, intercept, r_value, p_value, std_err = sp.stats.linregress(std,MPS)
#print('Slope=' + str(slope) + ', intercept=' + str(intercept) + ', r_value=' + str(r_value) + ', p_value=' + str(p_value))

plt.ylabel('Aggregate Saving Rate')
plt.xlabel('Std Dev of Perm. Income Shock')
plt.title('Change in Savings Following Increase in Permanent Income Uncertainty')
plt.ylim(0.045,0.075)
plt.scatter(std,ASR)
# Draw the linear fitted line
m, b = np.polyfit(std, ASR, 1)
plt.plot(std, m*std + b, '-')
slope, intercept, r_value, p_value, std_err = sp.stats.linregress(std,ASR)
print('Slope=' + str(slope) + ', intercept=' + str(intercept) + ', r_value=' + str(r_value) + ', p_value=' + str(p_value))