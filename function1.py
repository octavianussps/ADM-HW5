#!/usr/bin/env python
# coding: utf-8

# # Functionality 1 - Find the Neighbours!
# 
# Take input:
# 
# a node v
# 
# One of the following distances function: 
# 
# t(x,y), d(x,y) or network distance .
# 
# a distance threshold d.
# 
# Implement an algorithm (using proper data structures) that returns :
# 
# the set of nodes at distance <= d from v, corresponding to v’s neighborhood.

# In[1]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


# # Read DS

# # Distance :

# In[2]:


df_D = pd.read_csv('./DATA/Distance_graph',header=0, sep=',',
                index_col=None, encoding = "ISO-8859-1")
df_D.columns=["Distance"]
df_D.drop([0,1,2,3,4,5],inplace=True)
df_D.reset_index(drop=True)
df_D.head(10)


# In[3]:


df_D=df_D['Distance'].str.split(" " , expand=True)
df_D.columns=['a','Node1','Node2','Distance']
df_D['Node1']=pd.to_numeric(df_D['Node1'])
df_D['Node2']=pd.to_numeric(df_D['Node2'])
df_D['Distance']=pd.to_numeric(df_D['Distance'])
df_D=df_D.reset_index(drop=True)
df_D.head(4)


# In[4]:


df_D.shape


# In[5]:


df_T = pd.read_csv('./DATA/Travel_time_graph',header=0, sep=',',
                index_col=None, encoding = "ISO-8859-1")
df_T.columns=["Distance"]
df_T.drop([0,1,2,3,4,5],inplace=True)
df_T.head(5)


# In[6]:


df_T=df_T["Distance"].str.split(" " , expand=True)
df_T.columns=['a','Node1','Node2','Distance']
df_T['Node1']=pd.to_numeric(df_T['Node1'])
df_T['Node2']=pd.to_numeric(df_T['Node2'])
df_T['Distance']=pd.to_numeric(df_T['Distance'])
df_T=df_T.reset_index(drop=True)
df_T.head(5)


# In[7]:


df_T.shape


# # Search Neighbours:
# 

# In[49]:


node= int(input("Enter node numbner "))
distance_type=input("Enter one of this Distance type: Time or Physical ")
threshold=int(input("Enter your threshold "))  
#The idea is to iterate all direct neighbour from node v
#and check the values Corresponding to the input
#and we recursively subtract the new distance from initial threshold 
#to check the neighbours of the initial node's neighbour's Satisfying the threshold
def search_Neighbours(node, distance_type, threshold): 
    if distance_type == "Physical":
        data = df_D
    elif distance_type == "Time":
        data=df_T   
    out_df = data[(data['Node1'] == node) & (data['Distance'] <= threshold)]

    Neighbours=list(out_df.Node2.values)
        #print(Neighbours)
    for j in Neighbours:
        neigh_dis = data[(data['Node1'] == node) & (data['Node2'] == j)]
        neigh_dis=int(neigh_dis['Distance'])
        out_df=pd.concat([out_df,search_Neighbours(j, distance_type, threshold-neigh_dis)])
    return out_df

out_df=search_Neighbours(node,distance_type,threshold)
out_df.head(7)


# In[50]:


all_Neighbours = list(set(out_df['Node2'].values))
all_Neighbours


# In[51]:
   


# In[ ]:





# # The map 

# In[52]:


import folium
from folium import plugins
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[53]:


df_c = pd.read_csv('./DATA/Coordinates',header=0, sep=',',
                index_col=None, encoding = "ISO-8859-1")
df_c.columns=["Coordinates"]
df_c.drop([0,1,2,3,4,5],inplace=True)
df_c.head(3)


# In[54]:


df_c=df_c.Coordinates.str.split(" " , expand=True)
df_c.columns=['v','Node_ID','long','lat']
df_c['Node_ID']=pd.to_numeric(df_c['Node_ID'])
df_c['long']=pd.to_numeric(df_c['long'])
df_c['lat']=pd.to_numeric(df_c['lat'])
df_c=df_c.reset_index(drop=True)
df_c.head(5)


# In[55]:


df_c['lat']=df_c['lat']/1000000
df_c['long']=df_c['long']/1000000


# In[56]:


df_c.head(4)


# In[57]:


out_df_cor=pd.merge(df_c,out_df, right_on="Node2", left_on='Node_ID' )


# In[58]:


out_df_cor.head(5)


# In[59]:


out_df_cor=out_df_cor.drop_duplicates()


# In[60]:


out_df_cor


# In[61]:


lat=list(df_c['lat'].values)
long=list(df_c['long'].values)


# In[62]:


m = folium.Map(location=[np.mean(lat),np.mean(long)],zoom_start=6)


# In[63]:


m


# In[64]:


# plotting the starting node and node within the threshold
for i in range(len(out_df_cor)):
    folium.CircleMarker(location=[lat[i],long[i]],
                        radius=5,
                        popup=['Node_ID'],
                        fill_color="#3db7e4",
                       ).add_to(m)


# In[65]:


m
#the map is howin in the .ipynb file 


# In[ ]:




