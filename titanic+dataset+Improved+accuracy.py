
# coding: utf-8

# In[78]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[79]:

dataset = pd.read_csv("C:/Users/ssn/Documents/Machine Learning/Logistic Regression/Titanic.csv")


# In[80]:

dataset.head()


# In[81]:

dataset = dataset.drop(['Name','Ticket','PassengerId','Cabin','Parch','SibSp'], axis = 1)


# In[82]:

dataset.head()


# In[83]:

dataset = pd.get_dummies(dataset, columns = ['Sex', 'Embarked'])


# In[84]:

dataset.head()


# In[85]:

import seaborn as sns


# In[86]:

dataset['Age'].plot(kind="box")


# In[87]:

dataset['Age'] = dataset['Age'].fillna(np.mean(dataset['Age']))


# In[88]:

dataset.isnull().sum()


# In[89]:

dataset.head()


# In[93]:

x = dataset.iloc[:, 1:9]


# In[94]:

y = dataset.iloc[:,: 1]


# In[95]:

x.head()


# In[96]:

y.head()


# In[97]:

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 1/3)


# In[98]:

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[99]:

dataset.skew(axis = 0, skipna = True)


# In[100]:

sns.boxplot(dataset['Fare'])


# In[101]:

dataset_Fare_Q1 = dataset['Fare'].quantile(0.25)
dataset_Fare_Q3 = dataset['Fare'].quantile(0.75)
dataset_Fare_IQR =dataset_Fare_Q3 - dataset_Fare_Q1
print(dataset_Fare_Q1,dataset_Fare_Q3,dataset_Fare_IQR)
dataset = dataset[~((dataset.Fare<(dataset_Fare_Q1-1.5*dataset_Fare_IQR))|(dataset.Fare>(dataset_Fare_Q3+1.5*dataset_Fare_IQR)))]
sns.boxplot(x=dataset['Fare'])


# In[56]:

dataset_Embarked_Q_Q1 = dataset['Embarked_Q'].quantile(0.25)
dataset_Embarked_Q_Q3 = dataset['Embarked_Q'].quantile(0.75)
dataset_Embarked_Q_IQR =dataset_Embarked_Q_Q3 - dataset_Embarked_Q_Q1
print(dataset_Embarked_Q_Q1,dataset_Embarked_Q_Q3,dataset_Embarked_Q_IQR)
dataset = dataset[~((dataset.Embarked_Q<(dataset_Embarked_Q_Q1-1.5*dataset_Embarked_Q_IQR))|(dataset.Embarked_Q>(dataset_Embarked_Q_Q3+1.5*dataset_Embarked_Q_IQR)))]
sns.boxplot(x=dataset['Embarked_Q'])


# In[102]:

dataset.skew(axis = 0, skipna = True)


# In[103]:

from sklearn.linear_model import LogisticRegression


# In[104]:


classifier = LogisticRegression()
classifier.fit(x_train,y_train)


# In[105]:

y_pred = classifier.predict(x_test)


# In[106]:

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# In[107]:

classifier.score(x_test, y_test)


# In[108]:


from sklearn.metrics import precision_recall_fscore_support


# In[109]:

all=precision_recall_fscore_support(y_test, y_pred, average='macro')


# In[110]:

all


# In[111]:

from sklearn.naive_bayes import GaussianNB

classifier=GaussianNB()
classifier.fit(x_train,y_train)


# In[112]:

all=precision_recall_fscore_support(y_test, y_pred, average='macro')


# In[113]:

all


# In[114]:

from sklearn.tree import DecisionTreeClassifier


# In[115]:

classifier = DecisionTreeClassifier()
classifier.fit(x_train,y_train)


# In[116]:

all=precision_recall_fscore_support(y_test, y_pred, average='macro')


# In[117]:

all


# In[118]:

from sklearn.svm import SVC


# In[119]:

classifier = SVC(kernel='linear')
classifier.fit(x_train,y_train)


# In[120]:

all=precision_recall_fscore_support(y_test, y_pred, average='macro')


# In[121]:

all


# In[ ]:



