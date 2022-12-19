#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
from os import system
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# # Web scraping data using pandas

# In[1]:


import pandas as pd

url = 'https://www.basketball-reference.com/leagues/NBA_2023_per_game.html'
html = pd.read_html(url, header = 0)
df2023 = html[0]

df2023[60:65]


# In[1]:


# df[0] simply refers to the first dataframe: This function searches for <table> elements and only for <tr> and <th> rows 
# and <td> elements within each <tr> or <th> element in the table. <td> stands for “table data”.


# As there're some rows repeating the header, we remvoe it below:

# In[2]:


#Data cleaning
raw = df2023.drop(df2023[df2023.Age == 'Age'].index)
raw[58:63]


# In[9]:


raw.shape


# In[11]:


raw.info()


# In[12]:


# Check for porpotion of missing values
missing_values = raw.isnull().sum()
missing_values_per = (raw.isnull().sum() / raw.isnull().count())
pd.concat([missing_values, missing_values_per],axis=1, keys = ['Count_of_Missing','Percentage'])
r = pd.concat([missing_values, missing_values_per],axis=1, keys = ['Count_of_Missing','Percentage']).reset_index()
r


# * FG%	Field Goal Percentage
# * 3P%	FG% on 3-Pt FGAs.
# * eFG%	Effective Field Goal Percentage (Note: This statistic adjusts for the fact that a 3-point field goal is worth one more point than a 2-point field goal.)
# * 2P%	FG% on 2-Pt FGAs.
# * FT%	Free Throw Percentage
# # Fill missing value with 0

# In[13]:


df = raw.fillna(0)
df.isnull().sum()


# In[16]:


#drop ranking
df = df.drop(['Rk'], axis=1) #axis=1 is to drop by column
df.tail()


# In[4]:


#write to csv
df.to_csv(r'C:/Users/Python/portfolio/nba2023.csv', index = False)


# In[5]:


#read data back in
# address = "C:/Users/Python/portfolio/"
df = pd.read_csv('nba2023.csv',header=0)


# In[37]:


pd.reset_option('display.max_rows')


# In[38]:


# if we want to examine the whole dataset
pd.set_option('display.max_columns', df.shape[0]+1)#or display.max_rows
df


# In[ ]:


# pd.set_option('display.max_rows', 10);df


# In[39]:


df.dtypes


# # Show specific data types in dataframe

# In[40]:


df.select_dtypes(include=['number'])


# In[41]:


df.select_dtypes(include=['object'])


# ## Q1. Which player scored the most Points (PTS) Per Game?

# In[42]:


df[df.PTS == df.PTS.max()]


# ### what team is the player from?

# In[43]:


df.Tm[df.PTS == df.PTS.max()]


# In[44]:


#method2
PlayerMaxPoints = df[df.PTS == df.PTS.max()]
PlayerMaxPoints.Tm


# ### Which position is the player playing as?

# In[45]:


df.Pos[df.PTS == df.PTS.max()]


# In[ ]:


### How many games did the player played in the season?


# In[46]:


PlayerMaxPoints.G


# ## Q2.Which player scored more than 20 Points (PTS) Per Game?

# In[49]:


df[df.PTS >20]


# ## Q3.Which player had the highest 3-Point Field Goals Per Game (3P) ?

# In[7]:


df[df['3P'] == df['3P'].max()]


# ## Q4.Which player had the highest Assists Per Game (AST) ?

# In[8]:


df[df['AST'] == df['AST'].max()]


# ## Q5. Which player scored the highest (PTS) in the Los Angeles Lakers?
# 
# Use groupby() and get_group()

# In[9]:


LAL = df.groupby('Tm').get_group('LAL')
LAL[LAL.PTS == LAL.PTS.max()]


# ## Q6.Of the 5 positions, which position scores the most points?

# In[10]:


df.groupby('Pos').PTS.describe()


# Most scored is made by position PG (9.7585), but they are roughly similar.

# In[ ]:


# if there's players double-playing different position, we could specify the postion we want
# positions = ['C','PF','SF','PG','SG']
# POS = df[df['Pos'].isin(positions)]


# # Data Visualization
# ## 1. Histogram: pandas.hist()
# Built in in pandas package

# In[24]:


#method2. showing some histogram plots with subquery
PTS_byPOC = df.loc[:,['Pos', 'PTS']];PTS_byPOC.head(10)


# In[26]:


#Pandas's built-in: hist()
PTS_byPOC['PTS'].hist(by=PTS_byPOC['Pos'], layout=(1,5), figsize=(16,2))
#adjust the layout to be 1 row, 5 columns, figure size to be 16 by 2


# ## 2. Seaborn + plt.hist()

# In[29]:


#using seaborn for visualization
import seaborn as sns
import matplotlib.pyplot as plt

g = sns.FacetGrid(PTS_byPOC, col="Pos")#col="Pos" is to separate by position
g.map(plt.hist, "PTS");


# ## 3. pandas.boxplot() + sns.boxplot()

# In[30]:


# Boxplot: Pandas built-in function
PTS_byPOC.boxplot(column='PTS', by='Pos')#separated by position


# We could see that PG has the highest median (the green line), and PG, SG have wide IQR. But with boxplot, we cannot tell if the distribution of the data is centered about two points or pretty much spread even across the data range.

# In[31]:


#seaborn
sns.boxplot(x = 'Pos', y = 'PTS', data = PTS_byPOC) 


# ### 3.2 Draw a categorical scatterplot using jitter to reduce overplotting.

# In[32]:


#seaborn with distribution
sns.boxplot(x = 'Pos', y = 'PTS', data = PTS_byPOC) 
sns.stripplot(x = 'Pos', y = 'PTS', data = PTS_byPOC,
              jitter=True, #jitter=True so that it would randomize the number wouldn't overlap that much
              marker='o',
              alpha=0.8, 
              color="black")


# ## 4. Heatmap: Compute the correlation matrix

# In[34]:


corr = df.corr()
corr


# In[36]:


fig, ax = plt.subplots(figsize=(7,5))
sns.heatmap(corr, square=True)


# The white diagnoial line indicates a correlation, coefficient of 1, and red as 0.5
# 
# mask=dataframe or = True: If passed, data will not be shown in cells where mask is True. Cells with missing values are automatically masked. If my goal was to build a model to predict the brand of a car based on its unique features, I may have to remove one or more of these three variables in order to optimize my model and prevent multicolinearity.
# 

# In[48]:


#Mask diagonal half of heat map (Diagonal correlation matrix)
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True #Return the indices for the upper-triangle of arr
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(16, 15))
    ax = sns.heatmap(corr, mask=mask, vmax=1, square=True, annot=True, linewidth=.5)


# ## 5. Scatter Plot

# In[41]:


# subset to numeric only variables 
number = df.select_dtypes(include=['number']);
number.iloc[:4,:5]


# In[42]:


#subset to the first 5 and pts columns
selections = ['Age', 'G', 'GS', 'MP', 'FG', 'PTS']
df5 = df[selections]
df5


# In[43]:


g = sns.PairGrid(df5)
g.map(plt.scatter);


# In[ ]:


# g = sns.PairGrid(number)
# g.map(plt.scatter);


# Reference:
# 1. [GeeksforGeeks. Cat command](https://www.geeksforgeeks.org/using-the-cat-command-in-python/)
# 2. [GeeksforGeeks. Add columns to df](https://www.geeksforgeeks.org/add-column-names-to-dataframe-in-pandas/)
# 3. [Digitalocean. 3 ways to easily create a subset of Python](https://www.digitalocean.com/community/tutorials/create-subset-of-python-dataframe)
# 4. [Justinsighting. How to interpret boxplot](https://justinsighting.com/how-to-interpret-box-plots/#:~:text=A%20box%20plot%20gives%20us%20a%20basic%20idea,of%20the%204%20sections%20of%20the%20box%20plot.)
# 5. [Data Professor. Exploratory Data Analysis in Python using pandas](https://www.youtube.com/watch?v=9m4n2xVzk9o&list=PLtqF5YXg7GLltQSLKSTnwCcHqTZASedbO&index=14)
# 6. [Seaborn. sns.stripplot documentation](https://seaborn.pydata.org/generated/seaborn.stripplot.html)
# 7. [Medium. 5min Heatmap](https://medium.com/5-minute-eda/5-minute-eda-correlation-heatmap-b57bbb7bae14)
# 8. [Seaborn. sns.heatmap documentation](https://seaborn.pydata.org/generated/seaborn.heatmap.html)
