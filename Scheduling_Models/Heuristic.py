#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 11:28:26 2021

Heuristic rule

@author: fanjia
"""

import numpy as np
import pandas as pd
#import plotly.express as px

import gurobipy as gp
from gurobipy import Model, GRB, quicksum, max_
from random import randint, seed

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

import time

start_time = time.time()

# daily machine available hours
CT_hours=pd.read_excel("CThours_endOfWarmup_heuristics.xlsx")
LINAC_hours=pd.read_excel("Linachours_endOfWarmUp_heuristics.xlsx")
df_partial = pd.read_excel("partialBooked_endOfWarmup.xlsx")

# read sample input instance
instance_all=pd.read_excel("ins_2018_2020_heuristics.xlsx")
df_CT_wait = pd.read_excel('heuristics_CTWait_mean_GP.xlsx')
df_total_wait = pd.read_excel('heuristics_totalWait_mean_GP.xlsx')

def checkConsecutive(Linac_hours, start, unit, numFracs, duration):
    ready=True
    remainDays = 26 - start
    if remainDays > numFracs:
        for i in range(0, numFracs):
            if (Linac_hours[Linac_hours['day']==start+i][unit] >= duration - apptDiff).bool():
                ready = True
            else:
                ready = False
                break
    else: # if numFracs > remainDays
        for i in range(0, remainDays):
            if (Linac_hours[Linac_hours['day']==start+i][unit] >= duration - apptDiff).bool():
                ready = True
            else:
                ready = False
                break
        
    return ready

ins_target = instance_all[(instance_all['CreatedDate'] >= pd.to_datetime("2019-11-01"))&
                          (instance_all['CreatedDate'] < pd.to_datetime("2020-03-01"))] # change the run period here

# output record file
output_file = "record_2019_2020_80p_heuristics_MIP_addp.xlsx"
#preTxDays_target = 'preTxDays_80'
apptDiff = 10
objVal = 0

df_record = pd.DataFrame(columns=['MRN','Category','Intent', 'CreatedDate','ModelWaitTime','CTUnit','LinacUnit','TxDuration','numFracs'])
df_record_CT = pd.DataFrame(columns=['MRN','Category','Intent', 'CreatedDate','CTWait','CTUnit'])

#machine_hours_booked = pd.DataFrame(columns = ['Unit','totalMinutes'])
#machine_hours_booked['Unit']=[1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17]
#machine_hours_booked['totalMinutes'] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#machine_hours_booked.set_index('Unit', inplace=True)

# loop starts...............................................
all_dates = ins_target['CreatedDate'].unique()
#all_dates = instance_all['CreatedDate'].unique()
#all_dates = ['2019-12-02T00:00:00.000000000']
all_dates.sort()

for date in all_dates:
    print("NEW INSTANCE......................................................")
    print("This date is "+ str(date))
    instance = instance_all[instance_all['CreatedDate']==date]
    
    # iterate patients in a day by the order of arrival
    for index, row in instance.iterrows(): # for every patients
        
        # this patient's info
        MRN = row['MRN']
        print('MRN is ....................', MRN)
        cate = row['Category_x']
        intent = row['Intent1']
        createdDate = row['CreatedDate']
        
        earliestCTDay = int(df_CT_wait[(df_CT_wait['sitegroup']==row['SiteGroup'])&
                                   (df_CT_wait['category']==row['Category_x'])&
                                   (df_CT_wait['intent']==row['Intent1'])]['CTWait'])
        earliestStartDay = int(df_total_wait[(df_total_wait['sitegroup']==row['SiteGroup'])&
                                   (df_total_wait['category']==row['Category_x'])&
                                   (df_total_wait['intent']==row['Intent1'])]['totalWait'])
                               
                               
        cap_CT=row['CT']
        cap_linac=row['linacs']
        list_linacs = list(map(int, list(cap_linac.split(','))))
         # set of lists of capable linac machines
        
        
        # book CT .....................................................................................................
        CTbooked = 0
        CTdelay = 0  
        
        while CTbooked==0:
            if (CT_hours[CT_hours['day']==earliestCTDay][cap_CT]>=row['SimApptDuration']).bool():
                CTbooked = 1
                ct = cap_CT # the CT simulator booked on
                print('CT booked.')
                # update CT available hours
                current = CT_hours[CT_hours['day']==earliestCTDay][cap_CT]
                new = current - row['SimApptDuration']
                CT_hours.loc[CT_hours['day']==earliestCTDay, [cap_CT]] = new
                break
                
            else:
                earliestCTDay += 1
                CTdelay += 1
                
        # Book treatment .....................................................................................................
        trtBooked=0
        duration = row['TxApptDuration']
        numFracs = row['TxFracs']
        ptime = row['preTxDays_80']
        l=0
        earliestStartDay = earliestStartDay + CTdelay
        if earliestStartDay-earliestCTDay < ptime:
            earliestStartDay = int(earliestCTDay + ptime)
            print('add ptime')
        while trtBooked == 0:
            print('Earliest start date ', earliestStartDay, list_linacs[l])
            bookReady = checkConsecutive(LINAC_hours, earliestStartDay, list_linacs[l], numFracs, duration)
            avail = LINAC_hours[LINAC_hours['day'] == earliestStartDay][list_linacs[l]]
            #if ((avail>=row['TxApptDuration']).bool() and (bookReady)):
            if ((avail - row['TxApptDuration']>=0).bool() and bookReady):
               trtBooked = 1
               print('book right away.')
               break
                # if not bookready, consider alternative linacs
                                 
            elif l < len(list_linacs)-1: # else if there are other alternative linacs to check
                while l < len(list_linacs)-1:
                    l+=1
                    bookReady = checkConsecutive(LINAC_hours, earliestStartDay, list_linacs[l], numFracs, duration)
                    avail = LINAC_hours[LINAC_hours['day'] == earliestStartDay][list_linacs[l]]
                    if ((avail >= row['TxApptDuration']+0).bool() and (bookReady)):
                    # if the linac has enough time available, book tx appt
                        print('book on alternative linac ', list_linacs[l])
                        trtBooked = 1
                        break
                    else:
                        print('alternative linac also not available: unit ', list_linacs[l])
                        
            
            else: # else, start treatment on the next day
                if earliestStartDay <= 25:
                    earliestStartDay += 1
                    l=0
                    print('extend by one day. Earliest start date is ', earliestStartDay)
                    print('Check linac avail on new earliest start date ', earliestStartDay, list_linacs[l])
                    
                else: # earliestStartDate == 26
                    break
                
        # record treatments appointments and update linac hours..........................................................
        list_record=[]
        if trtBooked == 1: 
            linacBooked = list_linacs[l]
            wait = earliestStartDay
            list_record.append([MRN, cate, intent, createdDate, wait, ct, linacBooked, duration, numFracs])
            df_record = df_record.append(pd.DataFrame(list_record, columns=df_record.columns))
            # record of trt appointments
            
            # Book remaining treatment fractions
            if numFracs <= 26 - earliestStartDay: # if all fracs can be booked in this H
                for i in range(0, numFracs):
                    current = LINAC_hours[LINAC_hours['day'] == earliestStartDay+i][list_linacs[l]]
                    newlinactime = current - duration + apptDiff
                    print('booking Frac # ', i)
                    print('linac # ', list_linacs[l])
                    print('current remain time ', current)
                    print('duration of this appt ', duration)
                    print('new linac time ', newlinactime)
                    LINAC_hours.loc[LINAC_hours['day'] == earliestStartDay+i, [list_linacs[l]]] = newlinactime
                    # all fracs booked
                
            else: # remaining fractions > remaining number of days in H
                print('partially booked,...')
                remainDays = 26 - earliestStartDay
                remainFracs = numFracs - remainDays
                for i in range(0, remainDays):
                    current = LINAC_hours[LINAC_hours['day'] == earliestStartDay+i][list_linacs[l]]
                    newlinactime = current - duration + apptDiff
                    LINAC_hours.loc[LINAC_hours['day'] == earliestStartDay+i, [list_linacs[l]]] = newlinactime
                    # update the current schedule - linac time
                    
                    # record unbooked fractions in df_partial
                #remainingFracs = numFracs - remainDays
                list_partial=[]
                if remainFracs > 0:
                    list_partial.append([MRN, linacBooked, remainFracs, duration])
                    df_partial = df_partial.append(pd.DataFrame(list_partial, columns=df_partial.columns))
                    df_partial = df_partial.reset_index()
                    df_partial.drop(columns=['index'], inplace = True)
                
            
                
    # Roll forward one day
    ### roll forward for the next day - CTs
     # dropping first and last row
    CT_hours_copy = CT_hours
    CT_hours_copy[[2,3,4]] = CT_hours_copy[[2,3,4]] + 10
    CT_hours_copy = CT_hours_copy.iloc[1:-1]
       #reset the day number
    CT_hours_copy['day']=CT_hours_copy['day']-1
    
    # add two new rows: one for a new day, the other for the arbituary day
    df2 = pd.DataFrame([[25,200,200,150],
                        [26,1000,1000,1000]], columns=['day',2,3,4])
                                    
    CT_hours_updated=CT_hours_copy.append(df2, ignore_index=True)
    #CT_hours_updated.index = CT_hours_updated.index + 1 # resetting the index
        
    ### roll forward for the next day - linacs
    # dropping first and last row
    linac_hours_copy = LINAC_hours
    linac_hours_copy[[1,2,3,4,5,6,7,8,9,10,11,12,15,16,17]] = LINAC_hours[[1,2,3,4,5,6,7,8,9,10,11,12,15,16,17]] + 10
    linac_hours_copy = linac_hours_copy.iloc[1:-1]
    # reset the day number
    linac_hours_copy['day']=linac_hours_copy['day']-1
    # add two new rows: one for a new day, the other for the arbituary day
    #day25 = [515,560,455,375,520,530,460,460,545,550,520,525,530,570,510]
    day25 = [215,260,255,175,220,230,160,160,245,250,220,225,230,570,510]
            #[550]*16
    day26 = [1000]*15
    df2 = pd.DataFrame([[25]+day25,
                        [26]+day26], columns=['day',1,2,3,4,5,6,7,8,9,10,11,12,15,16,17])
    linac_hours_updated = linac_hours_copy.append(df2, ignore_index=True)
    #linac_hours_updated.index = linac_hours_updated.index + 1 # resetting the index
        
    ### book remaining sessions by updating linac hours on the new day 20................................................................
    # every time the machine hours is rolled forward, day 20 is newly added
    # for partially booked patients: on each new day 20,
    for index, row in df_partial.iterrows():
            # machine (linac) hours: -1*TxApptDur ( day 20 )
            # df_partial: FracsRemain -1
            # df_partial: drop row if FracsRemain == 0
            unit = row['Unit']
            MRN = row['MRN']
            TxDur = row['TxDur']
            linac_hours_updated.loc[25, unit] -= (TxDur - apptDiff) # this row is day 20
            #machine_hours_booked.at[linac,'totalMinutes'] += (TxDur - apptDiff)
            df_partial.loc[index, 'FracsRemain'] -= 1 # reduce the remaining fracs by 1
        
    #if row['FracsRemain']<=0: # drop from partial booked dataframe
     #       df_partial.drop(index, inplace = True)
    df_partial = df_partial[df_partial['FracsRemain']>0]
    
    # updated linac and CT hours used in model.........................................................................................
    CT_hours = CT_hours_updated
    LINAC_hours = linac_hours_updated

comptime = time.time() - start_time     
print("--- %s seconds ---" % comptime)

df_record.to_excel("record_2019_2020_80p_heuristics_MIP_addp.xlsx")


df_record = pd.read_excel('record_2019_2020_80p_heuristics_MIP_addp.xlsx')
df_timeframe = df_record[df_record['CreatedDate']>='2019-12-01']


import matplotlib.pyplot as plt
plt.figure(figsize=[9,5])
bins = range(0,24)

y = df_timeframe['ModelWaitTime']
plt.hist(y, bins = bins, alpha=0.5, edgecolor='black', align='left')

plt.xlabel('wait time')
plt.ylabel('frequency')
plt.title('Wait time distribution (heuristics based on MIP)')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.show()


print('mean: ', df_timeframe['ModelWaitTime'].mean())
print('std: ', df_timeframe['ModelWaitTime'].std())
print('wait time quantiles:\n', df_timeframe['ModelWaitTime'].quantile([.1, .5, .75]))
print('number of exceed: ', len(df_timeframe[df_timeframe['ModelWaitTime']>10]))
print('urgent: ', df_timeframe[df_timeframe['Category']=='Urgent 2']['ModelWaitTime'].mean())
print('urgent & palliative: ', df_timeframe[(df_timeframe['Category']=='Urgent 2')&
                                           (df_timeframe['Intent']=='Palliative')]['ModelWaitTime'].mean())
#print('objective value of sum z: ', z_obj)
print('total computational time: ', comptime) #2019-11 to 2020-02


    

    
        
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    