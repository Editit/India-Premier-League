
# coding: utf-8

# In[297]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
os.chdir("E:\Data\ipl")

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style ='whitegrid')
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


# In[187]:


#importing csv files
matches = pd.read_csv("matches.csv")
deliveries = pd.read_csv("deliveries.csv")


# In[188]:


matches.head(10)


# In[308]:


#Missing value Analysis
matches.shape
matches.isnull().sum()


# In[189]:


#dropping umpire3 column as there are all missing values
matches = matches.drop(labels='umpire3',axis=1)


# In[190]:


matches.isnull().sum()              


# In[191]:


#missing city values
missing_city = matches.loc[matches.city.isnull()]


# In[192]:


missing_city


# In[193]:


#imputing all the missing cities with Dubai
matches.city = matches.city.fillna('Dubai')


# In[194]:


#imputing missing winner to draw as results are tied
missing_winner = matches.loc[matches.winner.isnull()]
matches.winner = matches.winner.fillna('Draw')


# In[195]:


#checking for the missing umpire1 obervation
missing_ump1 = matches.loc[matches.umpire1.isnull()]
missing_ump1


# In[196]:


matches.umpire1 = matches.umpire1.fillna('VK Sharma')
matches.umpire2 = matches.umpire2.fillna('S Ravi')


# In[172]:


matches['team1']=le.fit_transform(matches['team1'])


# In[173]:


matches


# In[55]:


highest_mom = matches.player_of_match.value_counts()[:10]


# In[95]:


highest_mom


# In[97]:


#Most man of the matches
plt.figure(figsize=(14,9))
sns.barplot(x =highest_mom.index, y= highest_mom)
plt.xticks(rotation='vertical')


# In[81]:


plt.show()


# In[90]:


#highest team wins
highest_team_wins = matches.winner.value_counts()
plt.figure(figsize=(14,9))
plt.title("most successful ipl team")
plt.xticks(rotation='vertical')
sns.barplot(x=highest_team_wins.index, y=highest_team_wins)


# In[310]:


#mostly chosen after winning the toss
plt.figure(figsize=(12,6))
ax1 = sns.countplot(x='toss_decision',data = matches)


# In[206]:


df = matches[['team1','team2','city','toss_decision','toss_winner','venue','winner','season']]


# In[311]:


df.describe()


# In[314]:


#each city has the same venue, so they are carrying the same infomation
df = df.drop(labels = 'venue', axis = 1)


# In[212]:


matches.team1.unique()


# In[213]:


df.team1.unique()


# In[198]:


df.winner.unique()


# In[313]:


df.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                 'Sunrisers Hyderabad','Rising Pune Supergiants','Rising Pune Supergiant','Kochi Tuskers Kerala','Pune Warriors']
                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','RPS','KTK','PW'],inplace=True)

encode = {'team1': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
          'team2': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
          'toss_winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
          'winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13,'Draw':14}}
df.replace(encode, inplace=True)
df.head(10)


# In[219]:


df.city.unique()


# In[129]:


df.team1.value_counts()


# In[131]:


df['winner'].value_counts()


# In[235]:


df


# In[218]:


df.dtypes


# In[160]:


list = ['city','toss_decision','venue']


# In[ ]:


le = LabelEncoder()

for i in list:
    df[i]=le.fit_transform(df[i])
    


# In[141]:


df.dtypes


# In[150]:


df = df.drop(labels='venue',axis=1)


#  # Defining Classification model function

# In[300]:


#training = model.fit(X_train,Y_train)
#testing = model.predict(X_test)


# In[304]:


#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
  model.fit(data[predictors],data[outcome])
  predictions = model.predict(data[predictors])
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print('Accuracy : %s' % '{0:.3%}'.format(accuracy))
  kf = KFold(data.shape[0], n_folds=20)
  error=[]
  for train, test in kf:
    train_predictors = (data[predictors].iloc[train,:])
    train_target = data[outcome].iloc[train]
    model.fit(train_predictors, train_target)
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
  print('Cross-Validation Score : %s' % '{0:.3%}'.format(np.mean(error)))

  model.fit(data[predictors],data[outcome]) 


# In[291]:


X = df.values[:,0:4]
Y = df.values[:,5]


# In[298]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# In[307]:


outcome_var= ['winner']
predictor_var =  ['team1','team2','toss_winner','city','toss_decision','season']
model = DecisionTreeClassifier()

classification_model(model, df,predictor_var,outcome_var)


# In[266]:


model.predict(input)


# In[267]:


input= [1,7,7,10,1,2019]
input = np.array(input).reshape(1,-1)
output = model.predict(input)


# In[268]:


output

