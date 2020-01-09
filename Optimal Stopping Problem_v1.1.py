# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 12:34:46 2019

@author: Jimiwheels

Imagine an administrator who wants to hire the best secretary out of n rankable applicants for a position.
The applicants are interviewed one by one in random order. 
A decision about each particular applicant is to be made immediately after the interview. 
Once rejected, an applicant cannot be recalled. 
During the interview, the administrator can rank the applicant among all applicants interviewed so far.
Administrator is unaware of the quality of yet unseen applicants. 
What is the optimal strategy (stopping rule) to maximize the probability of selecting the best applicant?

"""

""" 
Author's assumptions:
The "fitness" of the applicants is distributed normally
The adminstrator knows statistics, but doesn't know the mean nor standard deviation of the candidate pool.
"""    

#import pandas
import pandas as pd
import numpy as np
import scipy.stats as Stats
import matplotlib.pyplot as plt


    
#Set = dict()
Candidates = dict()
CandActRank = dict()
# NoOfSims = number of simulations for each number of applicants
NoOfSims=10000

# of applicatants to evaluate is NoOfAppl
# assume applicant "fitness" is distributed normally
StartVal=10
EndVal=210
StepSize=10
LastEval=EndVal-StepSize

RL_PctResults = dict()
RL_CountResults = dict()

for NoOfAppl in range(StartVal,EndVal,StepSize):
    print(str(NoOfSims)+" of Smulations of "+str(NoOfAppl)+" applicants started")
    IndStdDev=20    # Assgined individual standard deviation
    IndMean=100     # Assigned Individual mean
    for SimX in range(0, NoOfSims):
        Candidates[SimX] = pd.Series(IndStdDev*np.random.randn(NoOfAppl)+IndMean)
        CandActRank[SimX]=Candidates[SimX].rank(ascending=0, method='min')
    dfCandidates=pd.DataFrame(Candidates)
    dfCandActRank=pd.DataFrame(CandActRank)         #Actual rank of candidate in each simulation
    #rename dataframe columns
    dfCandidates=dfCandidates.add_prefix('Sim_')
    
    #loop through the Candidates and rank them
    #initialize row lists of data I want capture at each interview
    RL_SmplMeanFit = dict()       #list of sample mean fitness @ each interview
    RL_SmplStdFit = dict()        #list of sample stdev of fitness @ each interview
    RL_Rank = dict()              #list of appliant rank @ each interview
    RL_Bernards = dict()          #list of Bernards Estimate of rank for RH limit
    RL_Calc_Pctl = dict()         #list of calculated rarity of applicant based on
                                  # sample mean and sample standard deviation
    
    for Applicant in range(2,NoOfAppl):
        #Select the head 1st entry to NoOfAppl
        dfCandHead=dfCandidates.head(Applicant)
        #print dfCandHead
        #Calculate running stats for each interview
        #Calculate current candidates average "fitness"
        dfCandAvg=dfCandHead.mean(axis=0)
        RL_SmplMeanFit[Applicant] = dfCandHead.mean(axis=0)
        #print "Candidate Average Fitness"
        #print RL_SmplMeanFit [Applicant]
        #print dfCandAvg
        #Calculate current candidates standard deviation of "fitness"
        dfCandStDev=dfCandHead.std(axis=0)
        RL_SmplStdFit[Applicant] = dfCandHead.std(axis=0)
        #print "Candidate Standard Deviation"
        #print dfCandStDev
        #Use pandas.DataFrame.rank to rank on "fitness" of header
        dfCandRank=dfCandHead.rank(ascending=0)
        dfCandRankTail=dfCandRank.tail(1)
        RL_Rank[Applicant] = dfCandRankTail
        #Calculate the Bernards estimate of maximum percentile of what's left
        # MR = (N-0.3)/(N+0.4)
        # Solved Bernard's for average of last two median ranks
        # (N^2-0.9*N-0.17)/(N^2-0.2*N-0.24)
        N = NoOfAppl-Applicant                      #number of applicants left
        #Avg of Bernards estimate of rank N & Bernards estimate rank of N-1
        RL_Bernards[Applicant] = (N**2-0.9*N-0.17)/(N**2-0.2*N-0.24)    
        #Calculate the percentile of the candidate
        dfCandTail=dfCandHead.tail(1)          #grabs current applicant "fitness"
        dfCandCalcStd=(dfCandTail-dfCandAvg)/dfCandStDev
        CandCalcCDF=Stats.norm.cdf(dfCandCalcStd)
        dfCandCalcCDF = pd.DataFrame(CandCalcCDF)
        srCandCalcCDF = dfCandCalcCDF.iloc[0,:]
        RL_Calc_Pctl[Applicant] = srCandCalcCDF
        
    #make dataframes of the running data calculated in the Applicant for loop
    dfRunSmplMean=pd.DataFrame(RL_SmplMeanFit).transpose()
    dfRunSmplStd=pd.DataFrame(RL_SmplStdFit).transpose()
    dfRunCalcPctile=pd.DataFrame(RL_Calc_Pctl).transpose()
    srBernards=pd.Series(RL_Bernards)
    srBernards = srBernards.rename('Bernards')
    
    #create dataframe for plots
    dfPlotDF = pd.concat([srBernards, dfRunCalcPctile], axis=1)
    dfPlotDF = dfPlotDF.add_prefix('Sim_')
    dfPlotDF.reset_index(level=0, inplace=True)
    dfPlotDF.rename(columns={'index': 'Applicant'})
    #Plot dataframes to compare calculated percentile w/ Bernard's max percentile
    # gca stands for 'get current axes'  Axes is the 'data area' object of a plot
    ax = plt.gca()
    dfPlotDF.plot(kind='line',x='index', y='Sim_Bernards',ax=ax)
    dfPlotDF.plot(kind='scatter',x='index', y='Sim_0',ax=ax, color='red', legend=True)
    dfPlotDF.plot(kind='scatter',x='index', y='Sim_1',ax=ax, color='purple', legend=True)
    dfPlotDF.plot(kind='scatter',x='index', y='Sim_2',ax=ax, color='green', legend=True)
    #dfPlotDF.plot(kind='scatter',x='index', y='Sim_3',ax=ax, color='orange', legend=True)
    ax.set_xlabel('Applicant Number')
    ax.set_ylabel('Calculated Percentile of Applicant N after interview N')
    plt.show()
    #plt.savefig('Running_Calculated_Applicant_Percentile_for_'+str(NoOfAppl)+'_Applicants.png')

    #Plot show process is working as intended
    #Create loop to collate data to determine statistics of method
    RL_ResultActRank = dict()
    RL_ResultCalcPct = dict()
    
    
    #Grab each column of data in dfRunCalcPctile
    for Dataset, srRunCalcPctile in dfRunCalcPctile.iteritems():
        #Dataset goes from 0 to last column number
        #Each Pandas Series will have the column number as it series name
        #For each row from index 2 to NoOfAppl
        for Applicant in srRunCalcPctile.iteritems():
            #print Applicant[0]              #Applicant number
            #print Applicant[1]              #Run Calculated Percentile
            if Applicant[1]>srBernards[Applicant[0]]:
                #print("Found a solution in Dataset "+str(Dataset)+ " !")
                #print("Applicant Number " +str(Applicant[0])+" Run Calculated Percentile "+str(Applicant[1])+" is > " + str(srBernards[Applicant[0]]))
                ApplicantM1=Applicant[0]-1
                ActualRank=dfCandActRank.loc[ApplicantM1,Dataset]
                #print("Applicant actual rank was "+str(ActualRank))
                #create dataframe of Results to evalute method
                RL_ResultActRank[Dataset] = dfCandActRank.loc[ApplicantM1,Dataset]
                #ResultPerRank[Dataset] = 
                RL_ResultCalcPct[Dataset] = Applicant[1]
                break
    srResultActRank=pd.Series(RL_ResultActRank)
    srResultActRank=srResultActRank.rename('Actual Rank')
    srResultCalcPct=pd.Series(RL_ResultCalcPct)
    srResultCalcPct=srResultCalcPct.rename('Calc Pctile')
    #print("Number of times Rank was chosen:")
    #print srResultActRank.value_counts()
    #print("Percentage of time Rank was chosen:")
    #print srResultActRank.value_counts('1')
    #srResult=srResultActRank.value_counts('1')
    #Create dictionary of %rank results for each No of Applicants
    RL_PctResults[NoOfAppl]=srResultActRank.value_counts('1')
    RL_CountResults[NoOfAppl]=srResultActRank.value_counts()
dfPctResults=pd.DataFrame(RL_PctResults)
#dfPctResults.reset_index(level=0, inplace=True)
#dfPctResults.rename(columns={'index': 'Applicant Chosen'})
dfCountResults=pd.DataFrame(RL_CountResults)
print("Done with "+str(NoOfSims)+" Simulations from "+str(StartVal)+ " applicants up to "+str(LastEval)+" in steps of "+str(StepSize))

#create print function to save files
#FigResult=plt.figure(figsize=(20,10))     #creates figure object FigResult
#axN=FigResult.add_subplot(1,1,1)         #creates axes object on figure
#subplot(nrows, ncols, index, **kwargs)   (111) can be used for single plot
dfCountPctResults=dfCountResults/NoOfSims
axN=dfCountPctResults.plot.bar(figsize=(20,10),title="% Time Applicant Rank was chosen - Average of "+str(NoOfSims)+" Simulations")
axN.set_xlim(right=4)
axN.set_xlabel('Applicant Chosen')
axN.set_ylabel('Percent time Applicant Rank Chosen')
plt.show
plt.savefig('Variable_Rule_OptimalStopping_AvgOf_'+str(NoOfSims)+'_Simulations_'+str(StartVal)+'_Thru_'+str(LastEval)+'_Applicants.png')
    
dfTotalChosen=dfCountResults.sum(axis = 0, skipna = True)
dfPctChosen=dfTotalChosen/NoOfSims                          #proof this method chooses candidate 100% of time
        