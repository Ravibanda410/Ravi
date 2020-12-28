# -*- coding: utf-8 -*-
"""
Created on Sun May 17 11:13:08 2020

@author: RAVI
"""
### Multinomial Regression ####
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

Mdata = pd.read_csv("C:/RAVI/Data science/Assignments/Module 10 Multinomial Regression/Dataset/mdata.csv")
Mdata.head(10)

Mdata.columns="sn","id","female","ses","schtyp","prog","read","write","math","science","honors"

Mdata1 = Mdata.drop(["sn","id"],axis=1)

Mdata1.describe()
Mdata1.prog.value_counts()

# Boxplot of independent variable distribution for each category of choice 
sns.boxplot(x="prog",y="read",data=Mdata1)
sns.boxplot(x="prog",y="write",data=Mdata1)
sns.boxplot(x="prog",y="math",data=Mdata1)
sns.boxplot(x="prog",y="science",data=Mdata1)


# Scatter plot for each categorical choice of car
sns.stripplot(x="prog",y="read",jitter=True,data=Mdata1)
sns.stripplot(x="prog",y="write",jitter=True,data=Mdata1)
sns.stripplot(x="prog",y="math",jitter=True,data=Mdata1)
sns.stripplot(x="prog",y="science",jitter=True,data=Mdata1)

# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(Mdata1,hue="prog") # With showing the category of each car choice in the scatter plot
sns.pairplot(Mdata1) # Normal

# Correlation values between each independent features
Mdata1.corr()


train,test = train_test_split(Mdata1,test_size = 0.2)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class="multinomial",solver="newton-cg").fit(train.iloc[:,4:7],train.iloc[:,3])

test_predict = model.predict(test.iloc[:,4:7]) # Test predictions
help(LogisticRegression)

# Test accuracy 
accuracy_score(test.iloc[:,3],test_predict) # 60%


train_predict = model.predict(train.iloc[:,4:7]) # Train predictions 
# Train accuracy 
accuracy_score(train.iloc[:,3],train_predict) # 59.3%
