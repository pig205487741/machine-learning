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