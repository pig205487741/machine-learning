# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 17:50:02 2019

@author: ohyax
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

titan = pd.read_csv("C:\\Users\\ohyax\\Desktop\\zz\\ML\\train.csv")
features = titan.drop(['Survived','Name','Cabin','Ticket','PassengerId'],axis=1)
features = pd.DataFrame(features)
target = pd.DataFrame(titan.Survived)

## use average Age to fillna
nulls =features.Age.isnull().sum()
ageave = features.Age.sum()/(len(features.Age)-nulls)
features.Age = features.Age.fillna(ageave)

## 用最多人下岸的 "S" fillna   turn C,Q,S to 0,1,2
features.Embarked = features.Embarked.fillna('S')
le = preprocessing.LabelEncoder()
le.fit(["C","Q","S"])
features.Embarked =le.transform(features.Embarked)

## turn "Sex" female to 0 male to 1
le = preprocessing.LabelEncoder()
le.fit(["female","male"])
features.Sex =le.transform(features.Sex)

##ramdon forest feature extraction
rf= RandomForestRegressor(random_state=1, max_depth=10)
rf.fit(features,np.ravel(target))
feature_importance = rf.feature_importances_
fn = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
indices = np.argsort(feature_importance[0:]) 

## drop feature_importances smaller than 0.1
cl =len(features.columns)
for i in range (0,cl):

    if feature_importance[i] < 0.1 :
        
        features = features.drop(features.columns[i],axis=1)
        features = features.iloc[:, [j for j, c in enumerate(features.columns) if j != i]]
        cl =len(features.columns)
    if i > cl :
        break 
        
##  圖示feature_importances
#plt.title('Feature Importances')
#plt.barh(range(len(indices)), feature_importance[indices], color='pink', align='center')
#plt.yticks(range(len(indices)), [fn[i] for i in indices])
#plt.xlabel('Relative Importance')
#plt.show()

#for i in range (0,7):
#    feature_importance = -np.sort((-feature_importance)  )
#    x=feature_importance[0:i].sum()
#    plt.scatter( i , x  , color='r')
    
#plt.ylabel('cumulative feature importance')
#plt.xlabel('numbers of features')
#plt.show()


## standardlize the data
features= preprocessing.StandardScaler().fit_transform(features)

##perform PCA
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(features)
principalDataframe = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2','PC3'])
## 圖示 PCA
#print (pca.explained_variance_ratio_ )
#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.xlabel('number of PCA components')
#plt.ylabel('cumulative explained variance')
#plt.show()

##fit GaussianNB
data=pd.concat([principalDataframe,target],axis=1)
data_principalDataframe = principalDataframe.values
data_target = target.values

validation_size = 1/3
seed = 3
x_train, x_test, y_train, y_test = train_test_split(data_principalDataframe, data_target, test_size=validation_size, random_state=seed)

model = GaussianNB()
model.fit(x_train,np.ravel(y_train))
predicted= model.predict(x_test)
## output Accuracy 
t=0
for i in range (0,len(x_test)):
    if y_test[i] == predicted[i]:
        t+=1
print("Accuracy : ", t/len(x_test) )
            





