#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 11:43:33 2021
MIP model on patient batching ___two weeks___

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
CT_hours=pd.read_excel("CThours_endOfWarmup_batch_2.xlsx")
LINAC_hours=pd.read_excel("Linachours_endOfWarmUp_batch_2.xlsx")
df_partial = pd.read_excel("partialBooked_batch_2.xlsx")

# read sample input instance
instance_all=pd.read_excel("ins_2020_part3_batching_2.xlsx")

ins_target = instance_all[(instance_all['CreatedDate'] >= pd.to_datetime("2019-12-01"))&
                          (instance_all['CreatedDate'] < pd.to_datetime("2020-03-01"))] # change the run period here

# output record file
output_file = "record_201912_202003_80p_Batching_2.xlsx"
preTxDays_target = 'preTxDays_80'
apptDiff = 10
objVal = 0





### created dataframe for recording
#df_partial = pd.DataFrame(columns=['MRN', 'Unit', 'FracsRemain','TxDur'])
df_record = pd.DataFrame(columns=['MRN','Category','Intent', 'CreatedDate','ModelWaitTime','CTUnit','LinacUnit','TxDuration','numFracs'])
machine_hours_booked = pd.DataFrame(columns = ['Unit','totalMinutes'])
machine_hours_booked['Unit']=[1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17]
machine_hours_booked['totalMinutes'] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
machine_hours_booked.set_index('Unit', inplace=True)

# loop starts...............................................
all_batches = ins_target['Batch2'].unique()

all_batches.sort()

for batch in all_batches:
    print("NEW INSTANCE......................................................")
    print("This batch is "+ str(batch))
    instance = instance_all[instance_all['Batch2']==batch]

    # Sets ........................................................................
    # set of patients:
    J = [j for j in range(len(instance['MRN']))]
    num_patients=len(J)
    
    # set of CT simulators:
    CTs = [c for c in [2,3,4]] 
    # set of LINAC machines:
    LINACs = [l for l in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17]]  #14
    
    # planning horizon: 10 days + 1 artificial day
    H = [d for d in range(1,32)]    # d is from 0 or from 1..?... one
    # The cosultation day in the horizon
    C = [j for j in instance['CreatedDay']] #used to calculate wait time
    
    p=[] # pretreatment days
    D1=[] # CT appointment durations
    D2=[] # linac appointment durations
    cap_linacs=[]
    cap_CTs=[]
    F=[]
    pr=[]
    tw = []
    for index, row in instance.iterrows():
       
        # j=index
        s_j = int(row['TxFracs'])
        F.append(s_j) #set of number of fracs for patients
        
        D1_j = row['SimApptDuration']
        D1.append(D1_j) # set of CT duration
        D2_j = row['TxApptDuration']
        D2.append(D2_j) # set of linac durations
        p_j = row[preTxDays_target]
        p.append(p_j) # set of req. pre-treatment days
        
        linac_j = row['linacs']
        list_linacs = list(map(int, list(linac_j.split(','))))
        cap_linacs.append(list_linacs) # set of lists of capable linac machines
        
        CT_j = row['CT']
        if isinstance(CT_j, str):
            list_CTs = list(map(int, list(CT_j.split(','))))
            cap_CTs.append(list_CTs) # set of assigned CTs
        else:
            cap_CTs.append([CT_j])
          
        pr_j = row['priority']
        pr.append(pr_j)
        
        tw_j = row['targetWait']
        tw.append(tw_j)
    
    ### CreatedGurubi model.......................................................Create gurobi model.............
    with gp.Env(empty=True) as env:
        env.setParam('LogToConsole', 0)
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as model:
            model = gp.Model("RT_Patient_Scheduling")
            model.setParam('TimeLimit', 5*60)
    # create a list of tuples containing every iteration of j,d,c
    x_tuple = [(j,d,c) for j in J for d in H for c in CTs]
    # create a list of tuples containing iterations of j,d,l,s_j
    y_tuple=[]
    p_list=[]
    for j in J:
        p_list = [(j,d,l,s) for d in H for l in LINACs for s in range(0,F[j])]
        y_tuple+=p_list
    # create a list of tuples containing iterations of j,l
    L_tuple = [(j,l) for j in J for l in LINACs]
    
    #Create decision variables
    x = model.addVars(x_tuple, vtype=GRB.BINARY, name="x_jdc")
    y = model.addVars(y_tuple, vtype=GRB.BINARY, name="y_jd1")
    z = model.addVars(num_patients, vtype=GRB.BINARY, name="z_j")
    
    L = model.addVars(L_tuple, vtype=GRB.BINARY, name="L_jl")
    
    d1 = model.addVars(num_patients, vtype=GRB.INTEGER, name="d_1j")
    d2 = model.addVars(num_patients, vtype=GRB.INTEGER, name="d_2j")
    
    O = model.addVars(num_patients, vtype=GRB.INTEGER, name="O_j")
    U = model.addVars(num_patients, vtype=GRB.INTEGER, name="U_j")
    W = model.addVars(num_patients, vtype=GRB.INTEGER, name="waitTime_j")
    
    # objective..................................................................................................................
    #model.setObjective(quicksum(z[j]*O[j]*pr[j] + W[j] for j in J), GRB.MINIMIZE)
    model.setObjective(quicksum(O[j]*pr[j] + pr[j]*W[j] for j in J), GRB.MINIMIZE)
    
    # 1. dummy constraints for target wait time
    model.addConstrs((W[j]==d2[j]-C[j] for j in J), name='wait_time_calc')
    model.addConstrs((tw[j] == W[j]-O[j]+U[j] for j in J), name='overage_underage')
    # 2. define z variables: if z=1, WaitTime > 14 days; if z=0, WaitTime <= 14 days
    # if z[j]==0, WaitTime[j]<=14
    model.addConstrs(((z[j]==0)>>(W[j]<=tw[j]) for j in J), name='wait_time_z_var')
    model.addConstrs(((z[j]==1)>>(W[j]>=tw[j]) for j in J), name='wait_time_z_var')
    
    # 3. Book first treatment appt on d2 for patient j
    # if v=0, patient j is not scheduled on day d; if v=1, patient j's 1st fraction is scheduled on day d 
    v_tuple = [(j,d) for j in J for d in H]  
    v = model.addVars(v_tuple, vtype=GRB.BINARY, name="v_jd")
    model.addConstrs((v[j,d]==0)>>(quicksum(y[j,d,l,0] for l in LINACs)<=0) for j in J for d in H)
    model.addConstrs((v[j,d]==1)>>(d2[j]==d) for j in J for d in H)
    v2 = model.addVars(v_tuple, vtype=GRB.BINARY, name="v2_jd") # same for the CT appointment
    model.addConstrs((v2[j,d]==0)>>(quicksum(x[j,d,c] for c in CTs)<=0) for j in J for d in H)
    model.addConstrs((v2[j,d]==1)>>(d1[j]==d) for j in J for d in H)
    
    # 4. Sufficient time inbetween CT Sim and LINAC first treatment
    model.addConstrs((d2[j]-d1[j] >= (p[j])*quicksum(x[j,d,c] for c in CTs for d in H) for j in J),
                     name='pretreat_duration')
    # 5. Only one CT appointment should be book for each patient
    model.addConstrs(
        (quicksum(x[j,d,c] for c in CTs for d in H) == 1 for j in J), name='one_CT')
    # 6. Only one first fraction should be booked for each patient
    for j in J:
        model.addConstrs(
            (quicksum(y[j,d,l,0] for l in LINACs for d in H) == 1 for j in J), name='one_first_linac')
    # 7. Each patient is assigned to one linac machine
    model.addConstrs((L[j,l]==quicksum(y[j,d,l,0] for d in H) for j in J for l in LINACs), name='define_L')
    model.addConstrs(
        (quicksum(L[j,l] for l in LINACs) == 1 for j in J), name='one_linac_per_patient')
    # 8. All treatment fractions are booked on the same linac machine patient j is assigned to
    for j in J:
        model.addConstrs(
            (quicksum(y[j,d,l,s] for d in H) <= L[j,l] for s in range(0,F[j]) for l in LINACs), name='on_same_linac')
    # 9. Patient j should only be treated by the linac that are capable
    for j in J:
        model.addConstrs(
            (y[j,d,l,s] == 0 for s in range(0,F[j]) for l in LINACs if l not in cap_linacs[j] for d in H), 
                                                                                            name='capable_linac')
    # schedule on the CT that is assigned
    for j in J:
        model.addConstrs(
            (x[j,d,c] == 0 for c in CTs if c not in cap_CTs[j] for d in H), name = 'capable_CT')
    # 10.1 total number of frac booked is less or equal to the num prescribed
    for j in J:
        model.addConstr(
            (quicksum(y[j,d,l,s] for d in H for l in LINACs for s in range(0,F[j])) <= F[j] ), name="num_frac"
        )
    # 10.2  treatments on consecutive days                
    for j in J:
        model.addConstrs(
            (
                y[j,d+1,l,s+1] == y[j,d,l,s] for l in LINACs for s in range(0,F[j]-1) for d in H[:-1]
             ), name='consecutive_days')
        
    # 11. machine working hours
    ### for CTs
    model.addConstrs(
        (quicksum(D1[j]*x[j,d,c] for j in J) <= CT_hours[CT_hours['day']==d][c] for d in H for c in CTs),
        name='CT_hours')
    model.addConstrs(
        (quicksum(D1[j]*x[j,d,c] for j in J) >= 0 for d in H for c in CTs),
        name='CT_hours_pos')
    # for linacs, both first sessions and the rest of the sessions
    #for j in J:
        #model.addConstrs(
            #(quicksum(D2[j]*y[j,d,l,s] for s in range(0,F[j])) <= LINAC_hours[LINAC_hours['day']==d][l]
                         #for d in H for l in LINACs), name='1_linac_hours')
    ### for linac: all patients on that machine on that day <= capacity
    model.addConstrs(
        (quicksum((D2[j] - apptDiff)*y[j,d,l,s] for j in J for s in range(1,F[j])) + quicksum((D2[j])*y[j,d,l,0] for j in J)
         <= LINAC_hours[LINAC_hours['day']==d][l]
        for d in H for l in LINACs), name = 'linac_hours')
    model.addConstrs(
        (quicksum((D2[j])*y[j,d,l,s] for j in J for s in range(0,F[j])) >= 0 for d in H for l in LINACs),
        name='linac_hours_pos')

    
    # 12. Domains:
    model.addConstrs((d1[j] >= 0 for j in J), name='domain_d1')
    model.addConstrs((d2[j] >= d1[j] for j in J), name='domain_d2')
    model.addConstrs((O[j] >= 0 for j in J), name='domain_O')
    model.addConstrs((U[j] >= 0 for j in J), name='domain_U')
    model.addConstrs((W[j] >= 0 for j in J), name='domain_W')
    #........................................................................................................................
    model.optimize()
    status = model.status
    if status == GRB.UNBOUNDED:
        print('The model cannot be solved because it is unbounded')
        #sys.exit(0)
    if status == GRB.OPTIMAL:
        print('The optimal objective is %g' % model.objVal)
        #for v in model.getVars():
            #print('%s %g' % (v.varName, v.x))       
        #model.printAttr('X') # print
        #sys.exit(0)
        
    if status != GRB.INF_OR_UNBD and status != GRB.INFEASIBLE:
        print('Optimization was stopped with status %d' % status)
        #sys.exit(0)
        
    if status == GRB.INFEASIBLE:
        # do IIS
        print('The model is infeasible; computing IIS')
        model.computeIIS()
        if model.IISMinimal:
            print('IIS is minimal\n')
        else:
            print('IIS is not minimal\n')
        print('\nThe following constraint(s) cannot be satisfied:')
        for c in model.getConstrs():
            if c.IISConstr:
                print('%s' % c.constrName)
    # Store solution in some variables.........................................................................................................
    if status != GRB.INFEASIBLE:
        sol_c = model.getAttr('X', x)
        sol_l = model.getAttr('X', y)
        sol_d2 = model.getAttr('X', d2)
        sol_w = model.getAttr('X', W)
        
        # non-zero variables
        x_jdc = [i for i in sol_c if sol_c[i] > 0.5]
        y_jdls = [i for i in sol_l if sol_l[i] > 0.5]
    
    ### update machine hours - CT ...........................................................................................
    CT_hours_copy = CT_hours
    CT_hours_copy['Day']=CT_hours_copy['day']
    CT_hours_copy.set_index('Day', inplace=True)
    for i in x_jdc:
        p = i[0]
        d = i[1]
        ct = i[2]   
        # ct appt duration
        time_c = instance.iloc[p]['SimApptDuration']    
        current = CT_hours[CT_hours['day']==d][ct]
        new = current - time_c    
        CT_hours[CT_hours['day']==d][ct] = new + 10 #reserve some CT time every day
    ### roll forward for the next day - CTs
    # dropping first and last row
    CT_hours_copy = CT_hours_copy.iloc[10:-1]
    #reset the day number
    CT_hours_copy['day']=CT_hours_copy['day']-10
    
    # add two new rows: one for a new day, the other for the arbituary day
    df2 = pd.DataFrame([[21, 200, 145, 75],
                        [22, 200, 145, 75],
                        [23, 200, 145, 75],
                        [24, 200, 145, 75],
                        [25, 200, 145, 75],
                        [26, 200, 145, 75],
                        [27, 200, 145, 75],
                        [28, 200, 145, 75],
                        [29, 200, 145, 75],
                        [30, 200, 145, 75],
                        [31,5000,5000,5000]], columns=['day',2,3,4])
                        # 20, 170, 235, 137
                        
                        
    CT_hours_updated=CT_hours_copy.append(df2, ignore_index=True)
    CT_hours_updated.index = CT_hours_updated.index + 1 # resetting the index
    
    ### update machine hours - Linacs........................................................................................
    linac_hours_copy = LINAC_hours
    linac_hours_copy['Day']=linac_hours_copy['day']
    linac_hours_copy.set_index('Day', inplace=True)
    
    linac_avail = pd.DataFrame([[1,18],[3,21],[4,23],[5,15],[6,12],[7,16],[8,21],[11,27],
                                [2,29],[9,25],[10,15],[15,17],[12,10],[16,16],[17,19],[14,28]],
                                columns = ['linacs','newMins'])
    linac_avail.set_index(['linacs'],inplace=True)
    
    for j in y_jdls:
        p = j[0]
        d = j[1]
        linac = j[2] 
        s = j[3]
        
        # Tx appt duration
        time_l = instance.iloc[p]['TxApptDuration'] 
        
        current = linac_hours_copy[linac_hours_copy['day']==d][linac]
                 #linac_hours_copy.loc[d][linac] # d-1 ??? LINAC_hours[LINAC_hours['day']==d][l]
        if s == 0:
            new = current - time_l  
            machine_hours_booked.at[linac,'totalMinutes'] += time_l

        else:
            new = current - (time_l - apptDiff)
            machine_hours_booked.at[linac,'totalMinutes'] += (time_l-apptDiff)
            
        '''print('linac unit: ', linac)
        print('current time available: ', current)
        print('remaining minutes: ', new)'''
        
        linac_hours_copy[linac_hours_copy['day']==d][linac] = new
        #linac_hours_copy.loc[d, linac] = new + 20
        #                                       linac_avail.at[linac, 'newMins'] # reserved some linac time every day
    
    ### roll forward for the next day - linacs
    # dropping first and last row
    linac_hours_copy = linac_hours_copy.iloc[10:-1]
    # reset the day number
    linac_hours_copy['day']=linac_hours_copy['day']-10
    # add two new rows: one for a new day, the other for the arbituary day
    day21 = [515,560,455,375,520,530,460,460,545,550,520,525,597,530,570,510] #[550]*16
    day22 = [515,560,455,375,520,530,460,460,545,550,520,525,597,530,570,510]
    day23 = [515,560,455,375,520,530,460,460,545,550,520,525,597,530,570,510]
    day24 = [515,560,455,375,520,530,460,460,545,550,520,525,597,530,570,510]
    day25 = [515,560,455,375,520,530,460,460,545,550,520,525,597,530,570,510]
    day26 = [515,560,455,375,520,530,460,460,545,550,520,525,597,530,570,510] #[550]*16
    day27 = [515,560,455,375,520,530,460,460,545,550,520,525,597,530,570,510]
    day28 = [515,560,455,375,520,530,460,460,545,550,520,525,597,530,570,510]
    day29 = [515,560,455,375,520,530,460,460,545,550,520,525,597,530,570,510]
    day30 = [515,560,455,375,520,530,460,460,545,550,520,525,597,530,570,510]
    arbday = [1000]*16 # day 31
    df2 = pd.DataFrame([[21]+day21,
                        [22]+day22,
                        [23]+day23,
                        [24]+day24,
                        [25]+day25,
                        [26]+day21,
                        [27]+day22,
                        [28]+day23,
                        [29]+day24,
                        [30]+day25,
                        [31]+arbday], columns=['day',1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17])
    linac_hours_updated=linac_hours_copy.append(df2, ignore_index=True)
    linac_hours_updated.index = linac_hours_updated.index + 1 # resetting the index
    
    ### Record Booked Patients.............................................................................................
    list_record=[]
    for i in y_jdls:
        p=i[0]
        d=i[1]
        l=i[2]
        s=i[3]
        MRN = instance.iloc[p]['MRN']
        cate = instance.iloc[p]['Category_x']
        intent = instance.iloc[p]['Intent1']
        duration = instance.iloc[p]['TxApptDuration']
        createdDate = instance.iloc[p]['CreatedDate']
        numFracs = instance.iloc[p]['TxFracs']
        createdDay = instance.iloc[p]['CreatedDay']
        wait = sol_d2[p] - createdDay
        ct = instance.iloc[p]['CT'] 
        linac = l
        if s == 0: # record the first treatment, if first tx is booked the patient is booked
            list_record.append([MRN, cate, intent, createdDate, wait, ct, linac, duration, numFracs])
    df_record = df_record.append(pd.DataFrame(list_record, columns=df_record.columns))
    
    # Record patients who are partially booked (have appt on arbituary day 26)............................................................

    list_partial = []
    for i in y_jdls:
        p=i[0]
        d=i[1]
        if d == 31:
            bookedFracs = i[3] - 1 # num of fracs alr booked
            remainingFracs = instance.iloc[p]['TxFracs'] - bookedFracs
            unit = i[2] # all fracs booked on the same unit?
            MRN = instance.iloc[p]['MRN']
            TxDur = instance.iloc[p]['TxApptDuration']
            
            if remainingFracs > 0:
                list_partial.append([MRN, unit, remainingFracs, TxDur])
            
    df_partial = df_partial.append(pd.DataFrame(list_partial, columns=df_partial.columns))
    df_partial = df_partial.reset_index()
    df_partial.drop(columns=['index'], inplace = True)
    
    ### book remaining sessions by updating linac hours on the new day 16 - 20................................................................
    # every time the machine hours is rolled forward, day 16 - 20 is newly added
    # for partially booked patients: on each new day 16 - 20,
    for index, row in df_partial.iterrows():
        # machine (linac) hours: -1*TxApptDur ( day 20 )
        # df_partial: FracsRemain -1
        # df_partial: drop row if FracsRemain == 0
        unit = row['Unit']
        MRN = row['MRN']
        remain = row['FracsRemain']
        TxDur = row['TxDur']
            
        if remain > 10:
            for i in range(0, 10):
                linac_hours_updated.loc[21+i, unit] -= (TxDur - apptDiff)
            #linac_hours_updated.loc[21, unit] -= (TxDur - apptDiff) # this row is day 26
            #linac_hours_updated.loc[22, unit] -= (TxDur - apptDiff)  # this row is day 27
            #linac_hours_updated.loc[23, unit] -= (TxDur - apptDiff)  # this row is day 28
            #linac_hours_updated.loc[24, unit] -= (TxDur - apptDiff)  # this row is day 29
            #linac_hours_updated.loc[25, unit] -= (TxDur - apptDiff) # this row is day 30
        
            machine_hours_booked.at[unit,'totalMinutes'] += (TxDur - apptDiff) * 10
            df_partial.loc[index, 'FracsRemain'] -= 10 # reduce the remaining fracs by 10
            
        if (remain <= 10)&(remain > 0):
            for i in range(0, remain):
                linac_hours_updated.loc[21+i, unit] -= (TxDur - apptDiff)
            
            machine_hours_booked.at[unit,'totalMinutes'] += (TxDur - apptDiff) * remain
            df_partial.loc[index, 'FracsRemain'] -= remain # reduce the remaining fracs by 5
        
        
        if row['FracsRemain']<=0: # drop from partial booked dataframe
            print('dropping 0 remains')
            df_partial.drop(index, inplace = True)
    
    df_partial = df_partial[df_partial['FracsRemain']>0]

    # updated linac and CT hours used in model.........................................................................................
    CT_hours = CT_hours_updated
    LINAC_hours = linac_hours_updated
    objVal += model.objVal
    
    

print("--- %s seconds ---" % (time.time() - start_time))
    
df_record.to_excel(output_file)
#df_partial.to_excel('partialBooked_high_ML.xlsx')
#LINAC_hours.to_excel('Linachours_high_ML.xlsx')
#CT_hours.to_excel('CThours_high_ML.xlsx')
#machine_hours_booked.to_excel('util_202003_202006_80p_Batching_2.xlsx')

import matplotlib.pyplot as plt
plt.figure(figsize=[9,5])
bins = range(0,24)

y = df_record['ModelWaitTime']
plt.hist(y, bins = bins, alpha=0.5, edgecolor='black', align='left')

plt.xlabel('wait time')
plt.ylabel('frequency')
plt.title('wait time distribution of 80p 2020')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.show()

y = ins_target[preTxDays_target]
plt.hist(y, bins = bins, alpha=0.5, edgecolor='black', align='left')

plt.xlabel('number of days')
plt.ylabel('frequency')
plt.title('preTxDays distribution of current ins')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.show()
