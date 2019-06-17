from sklearn.linear_model import LogisticRegression, LinearRegression
import csv
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

PATH = '../Anonymous_data/'

def load_data(path, filename):
    csvfile = open(path + filename, 'r', encoding='utf-8')
    dict_reader = csv.DictReader(csvfile)
    return dict_reader

def split_dataset(x,y):
    x_train = x[0:1800]
    x_test = x[1801:]
    y_train = y[0:1800]
    y_test = y[1801:]
    return x_train, y_train, x_test, y_test


class Logistic:
    def __init__(self, penalty,dual,C,solver,multi_class):
        self.clf = LogisticRegression(penalty=penalty, dual=dual, C=C,solver=solver,multi_class='auto',max_iter=2000)

    def train(self, X, y):
        self.clf.fit(X=X, y=y)
        return self.clf

    def predict(self, X):
        result = self.clf.predict(X=X)
        result = result.astyoe(np.float64)
        return result

    def visualization(self,X,y):
        return

class linear:
    def __init__(self):
        self.clf = LinearRegression()

    def train(self, X, y):
        self.clf.fit(X=X, y=y)
        return self.clf

    def predict(self, X):
        result = self.clf.predict(X=X)
        result = result.astyoe(np.float64)
        return result


filename = "data.csv"
data = load_data(path=PATH, filename=filename)
X = list()
y1 = list()
y2 = list()
poly = PolynomialFeatures(2)
for row in data:
    X.append([row['\ufeffT'],row['R']])
    num1 = int(float(row['T1'])*100)
    num2 = int(float(row['T2'])*100)
    y1.append(num1)
    y2.append(num2)
X = np.array(X)
y1 = np.array(y1)
y2 = np.array(y2)
X =X.astype(np.float64)
d= poly.fit_transform(X[:,1].reshape(-1,1))
X = np.append(d, X[:,0].reshape(-1,1),axis=1)

y1 = y1.astype(np.int64)
y2 = y2.astype(np.int64)
x_train, y_train, x_test, y_test = split_dataset(X,y1)
#lab_enc = preprocessing.LabelEncoder()
#y_encoded = lab_enc.fit_transform(y)

#Lr = Logistic(penalty='l1', dual=False, C=0.8, multi_class='auto',solver='newton-cg')
Lr = linear()
clf = Lr.train(X=x_train, y=y_train)
res = clf.predict(x_test)
#s= clf.predict_proba(x_test)
score = clf.score(x_train,y_train)
print(score)
coef_ = clf.coef_
intercept = clf.intercept_
print(coef_)
print(intercept)






