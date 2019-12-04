import math
import pandas as pd
import numpy as np

f = pd.read_csv('train.csv')
t = pd.read_csv('test.csv')

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

# 貝式分類處理
class statistics_processing:
    def __init__(self, vector_list):
        self.vectors = vector_list
    
    def covariance_matrix(self):
        self.X = np.dot(self.vectors, self.vectors.transpose())
        return self.X
    
    def determinant(self):
        return np.linalg.det(self.covariance_matrix())
    
    def Gaussian(self ,training_data):
        self.X = np.dot(training_data.transpose(),np.linalg.inv(self.covariance_matrix()))
        self.X = np.dot(self.X, training_data)
        expotential = math.exp((-1/2)*self.X)
        return (1/((self.determinant()**(1/2))*((2*math.pi)**np.shape(self.covariance_matrix())[0]/2)))*expotential

# 只有Age, SibSp, Parch, Fare適用數值方法，測試看看
# train看看
'''資料操作'''
survived = pre_processing(f).transfer_to_list('Survived')
for statement in [0,1]:
    D = {0 :549/(891+549), 1:(891-549)/(891+549) }
    target = f[f['Survived'] == statement]
    age = np.array(pre_processing(target).fulfill_missing('Age'))
    sibsp = np.array(pre_processing(target).fulfill_missing('SibSp'))
    parch = np.array(pre_processing(target).fulfill_missing('Parch'))
    fare = np.array(pre_processing(target).fulfill_missing('Fare'))
    data_array = np.array([age, sibsp, parch, fare]).transpose()
    Mean = [pre_processing(target).mean('Age'), pre_processing(target).mean('SibSp'), pre_processing(target).mean('Parch'), pre_processing(target).mean('Fare')]
    vector_list = np.array([age-Mean[0], sibsp-Mean[1],parch-Mean[2],fare-Mean[3]])
    print(statistics_processing(vector_list).Gaussian(np.array([f['Age'][0], f['SibSp'][0], f['Parch'][0], f['Fare'][0]]))*D[statement])

print(pre_processing(f).class_measurement())
print(pre_processing(f).missing_position('Age'))
print(pre_processing(f).transfer_to_list('Sex'))

        
