#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install colormath')
import numpy as np
import os
import sys
import pandas as pd
import seaborn as sns
import math
from os import listdir
import glob
import scipy


# In[ ]:


data = ('C:/Users/IsaacV/Documents/input')
os.chdir(data)
os.getcwd()


# In[ ]:


data_file1 = 'madeup.xlsx'
Image_1= pd.read_excel(data_file1,sheet_name= 0, index_col= None, columns=('AIS','BOUTON','R','G','B'))
Image_1 
I1= Image_1.iloc[0:, 2:]
I1, Image_1


# In[ ]:


dist=pd.DataFrame(scipy.spatial.distance.cdist(I1,I1, metric='euclidean'),columns=Image_1['BOUTON'].unique(), index=Image_1['BOUTON'].unique())
dist


# In[ ]:


dist.index.values[1]


# In[ ]:


dist.shape
type(dist.shape)
dist.shape[0]

len(dist)

thresh=.2
#key = pd.DataFrame(index=range(0),columns=['Name','List','R','G','B'])
#key = key.fillna(0)

unique_val = 0
key_data = {'Name': [unique_val], 'Bouton': [[dist.index.values[0]]], 'R': [I1.iloc[0,0]], 'G': [I1.iloc[0,1]], 'B': [I1.iloc[0,2]]}
key = pd.DataFrame(data=key_data)
#print(key)
unique_val += 1
unique_bool = False

for i in range(1,dist.shape[0]):
    unique_bool = False
    for j in range(0,(i)):
        if dist.iloc[i,j] < thresh:
            #fun code
            print("hi")
            if unique_bool == False:
                for k in range(0,key.shape[0]):
                    if any(dist.index.values[j] in s for s in key.iloc[k,1]):
                        key.iloc[k,1].append(dist.index.values[i])
            unique_bool = True
    if unique_bool==False:
        key_data = {'Name': [unique_val], 'Bouton': [[dist.index.values[i]]], 'R': [I1.iloc[i,0]], 'G': [I1.iloc[i,1]], 'B': [I1.iloc[i,2]]}
        unique_val += 1
        key = key.append(pd.DataFrame(data=key_data))
        #print(temp_df)
print(key)


# In[ ]:



prev=1
value=-1
first_time = True
temp_list = []

for i in range(0,Image_1.shape[0]):
    for k in range(0,key.shape[0]):
        if any(Image_1.iloc[i,1] in s for s in key.iloc[k,1]):
            value=k
    
    if Image_1.iloc[i,0] != prev:
        if first_time:
            #temp_list = [value]
            transform_data = {'AIS': prev, 'Bouton_list': [temp_list], 'Size': len(temp_list)}#, 'Unique_Size': len(unique(temp_list))}
            transform = pd.DataFrame(data=transform_data)
            first_time = False
        else:
            transform_data = {'AIS': prev, 'Bouton_list': [temp_list], 'Size': len(temp_list)}#, 'Unique_Size': len(unique(temp_list))}
            transform = transform.append(pd.DataFrame(data=transform_data))
        temp_list = [value]
        prev = Image_1.iloc[i,0]
    else:
        temp_list.append(value)
        #print(temp_list)
        if i == (Image_1.shape[0]-1):
            transform_data = {'AIS': prev, 'Bouton_list': [temp_list], 'Size': len(temp_list)}#, 'Unique_Size': len(unique(temp_list))}
            transform = transform.append(pd.DataFrame(data=transform_data))
print(transform)
type(transform)
transform


# In[ ]:


print(key)


# In[ ]:


import random
import datetime

random.seed(datetime.datetime.now())

print(transform)


# In[ ]:


import pickle
#test = pickle.loads(pickle.dumps(transform))

#for i in range(0,10):
if True:
    random.seed(datetime.datetime.now())
    transform_iter = pickle.loads(pickle.dumps(transform))
    for j in range(0,transform_iter.shape[0]):
        for k in range(0,len(transform_iter.iloc[j,1])):
            transform_iter.iloc[j,1][k] = random.randint(0,key.shape[0]-1)
            
    print(transform_iter)


# In[ ]:


print(transform)


# In[ ]:


print(id(transform))
print(id(transform_iter))


# In[ ]:


print(key)
print(transform)


# In[ ]:


first_time = True
#logged = False
#logged_index = -1

for i in range(0,transform.shape[0]):
    if first_time == True:
        report_data = {'Bouton_Combination': [list(set(transform.iloc[i,1]))],'#AIS': 1, 'mean': -1, 'stdev': -1, 'pval': -1, 'padj': -1}
        report = pd.DataFrame(data=report_data)
        report.iloc[i,0].sort()
        first_time = False
    else:
        
        logged_index = -1
        transform.iloc[i,1].sort()
        for j in range(0,report.shape[0]):
            if report.iloc[j,0] == list(set(transform.iloc[i,1])):
                logged_index = j
        
        if logged_index == -1:
            report_data = {'Bouton_Combination': [list(set(transform.iloc[i,1]))],'#AIS': 1, 'mean': -1, 'stdev': -1, 'pval': -1, 'padj': -1}
            report = report.append(pd.DataFrame(data=report_data))
            report.iloc[i,0].sort()
        else:
            report.iloc[j,1] = report.iloc[j,1] + 1
        
        


# In[ ]:


print(report)


# In[ ]:


import pickle
import math
#test = pickle.loads(pickle.dumps(transform))

for i in range(0,10):
#if True:

    random.seed(datetime.datetime.now())
    transform_iter = pickle.loads(pickle.dumps(transform))
    for j in range(0,transform_iter.shape[0]):
        for k in range(0,len(transform_iter.iloc[j,1])):
            transform_iter.iloc[j,1][k] = random.randint(0,key.shape[0]-1)
    
    for k in range(0,transform_iter.shape[0]):
        first_time = True
        
        if first_time == True:
            report2_data = {'Bouton_Combination': [list(set(transform_iter.iloc[k,1]))],'#AIS': 1}
            report2 = pd.DataFrame(data=report2_data)
            report2.iloc[k,0].sort()
            first_time = False
        else:

            logged_index = -1
            transform_iter.iloc[k,1].sort()
            for j in range(0,report2.shape[0]):
                if report2.iloc[j,0] == list(set(transform_iter.iloc[k,1])):
                    logged_index = j

            if logged_index == -1:
                report2_data = {'Bouton_Combination': [list(set(transform_iter.iloc[k,1]))],'#AIS': 1}
                report2 = report2.append(pd.DataFrame(data=report2_data))
                report2.iloc[k,0].sort()
            else:
                report2.iloc[j,1] = report2.iloc[j,1] + 1
    
    index_list = []
    for k in range(0,report.shape[0]):
        found = False
        for m in range(0,report2.shape[0]):
            if report.iloc[k,0] == report2.iloc[m,0]:
                found = True
                index_list = index_list.append(m)
                if i == 0:
                    report.iloc[k,2] = report2.iloc[m,1]
                    report.iloc[k,3] = 0
                else:
                    temp_avg = report.iloc[k,2]
                    report.iloc[k,2] = ((i*report.iloc[k,2])+report2.iloc[m,1])/(i+1)
                    report.iloc[k,3] = math.sqrt(((i*((report.iloc[k,3]**2)+(temp_avg**2)+(report.iloc[k,2]**2)-(2*report.iloc[k,2]*temp_avg)))+((report2.iloc[m,1]-report.iloc[k,2])**2))/(i+1))
        if found == False:
            temp_avg = report.iloc[k,2]
            report.iloc[k,2] = ((i*report.iloc[k,2]))/(i+1)
            report.iloc[k,3] = math.sqrt(((i*((report.iloc[k,3]**2)+(temp_avg**2)+(report.iloc[k,2]**2)-(2*report.iloc[k,2]*temp_avg)))+((0-report.iloc[k,2])**2))/(i+1))
    
    index_list.sort()
    for k in range(0,report2.shape[0]):
        if (k in index_list == False):
            report_data = {'Bouton_Combination': [report2.iloc[k,0]],'#AIS': 0, 'mean': report2.iloc[k,1], 'stdev': 0, 'pval': -1, 'padj': -1}
            report = report.append(pd.DataFrame(data=report_data))
        
    

