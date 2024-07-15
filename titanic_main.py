# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import os
import copy
from sklearn.preprocessing import Binarizer
from sklearn.ensemble import RandomForestClassifier

path = """C:\\Users\\mathi\\Desktop\\Folders\\Project\\KaggleComp\\Titanic"""

rs = 23

df_gender_submission = pd.read_csv(path+'\\'+'gender_submission.csv', index_col='PassengerId')
df_train = pd.read_csv(path+'\\'+'train.csv', index_col='PassengerId')
df_test = pd.read_csv(path+'\\'+'test.csv', index_col='PassengerId')

y_train = df_train.pop('Survived')
x_train = copy.deepcopy(df_train)

x_test= copy.deepcopy(df_test)

#dropping some features, for pbly useless / too complex for now
x_train = x_train.drop(['Name','Ticket','Fare','Cabin','Embarked'],axis=1)
x_test = x_test.drop(['Name','Ticket','Fare','Cabin','Embarked'],axis=1)

#Binarizing some of the data
x_train.loc[x_train['Sex'] == 'male', 'Sex']=0
x_train.loc[x_train['Sex'] == 'female', 'Sex']=1

x_test.loc[x_test['Sex'] == 'male', 'Sex']=0
x_test.loc[x_test['Sex'] == 'female', 'Sex']=1


model = RandomForestClassifier(random_state = rs)

model.fit(x_train, y_train)

y_test = pd.DataFrame(model.predict(x_test), index=x_test.index)
y_test.to_csv(path+'\\submission.csv')
