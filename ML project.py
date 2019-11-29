import math
import pandas as pd
import numpy as np

f = pd.read_csv('train.csv')

# 先定義資料前處理方法
class pre_processing:
    def __init__(self, data):
        self.data = data
    
    def class_measurement(self):
        self.Category = list()
        for C in self.data:
            self.Category.append(C)
        return self.Category
            
    def missing_position(self, Class):
        self.result = list()
        for num in range(len(self.data[Class])):
            if self.data[Class].isnull().values[num]:
                self.result.append(num)    
        return self.result
    
    def transfer_to_list(self, Class):
        self.result = list()
        for ele in self.data[Class]:
            self.result.append(ele)
        return self.result
    
    def mean(self, Class):
        self.number = 0
        self.amount = 0
        big_list = self.transfer_to_list(Class)
        for num in big_list:
            if num >= 0:
                self.number += num
                self.amount += 1
        return self.number/self.amount
    
    def fulfill_missing(self, Class):
        self.fulfill = self.mean(Class)
        self.alist = self.transfer_to_list(Class)
        self.position = self.missing_position(Class)
        for P in self.position:
            self.alist[P] = self.fulfill
        return self.alist

# 只有Age, SibSp, Parch, Fare適用數值方法
survived = pre_processing(f).transfer_to_list('Survived')
age = np.array(pre_processing(f).fulfill_missing('Age'))
sibsp = np.array(pre_processing(f).fulfill_missing('SibSp'))
parch = np.array(pre_processing(f).fulfill_missing('Parch'))
fare = np.array(pre_processing(f).fulfill_missing('Fare'))
data_array = np.array([age, sibsp, parch, fare]).transpose()
Mean = [pre_processing(f).mean('Age'), pre_processing(f).mean('SibSp'), pre_processing(f).mean('Parch'), pre_processing(f).mean('Fare')]

class statistics_processing:
    def __init__(self, vector_list):
        self.vectors = vector_list
    
    def covariance_matrix(self):
        self.X = np.array()
        for num in range(len(self.vectors)):
            self.X = np.vstack(self.X, self.vectors[num])
        return self.X
