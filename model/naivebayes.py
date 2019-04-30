from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.model_selection import cross_validate
import csv
import pickle
PATH = '../UCI_dataset/'


class Data:
    value = list()
    label = list()

def trans0(level):
    if level=='g':
        return 1
    else:
        return 0

def trans1(t):
    t = str(t)
    if t =='BRICKFACE':
        return 0
    if t == 'SKY':
        return 1
    if t == 'FOLIAGE':
        return 2
    if t== 'CEMENT':
        return 3
    if t=='WINDOW':
        return 4
    if t=='PATH':
        return 5
    if t=='GRASS':
        return 6


def load_data(path, filename,f_type):
    d = None
    if f_type is 'txt':
        address = path+filename
        #d = np.loadtxt(path+filename)
        #d = np.loadtxt(path+filename, delimiter=',')
        #d = np.loadtxt(path + filename, converters={0 : lambda s: 0, 1:lambda  s:0})
        #d = np.loadtxt(path + filename, skiprows=1, converters={0: lambda s:0}, delimiter=',')
        #d = np.loadtxt(path + filename, converters={0: lambda s: 0}, delimiter=',',encoding='utf-8')
        #d = np.loadtxt(path + filename, converters={34: trans0}, delimiter=',', encoding='utf-8')
        #d = np.loadtxt(path + filename, delimiter=',', converters={0: trans1}, encoding='utf-8')
        #d = np.loadtxt(path + filename)
        d = np.loadtxt(path + filename, delimiter=',')
        return d
    elif f_type is 'csv':
        csvfile = open(path+filename, 'r',encoding='utf-8')
        dict_reader = csv.DictReader(csvfile)
        return dict_reader
    else:
        return d


class supportvectormachine(Data):
    data_set = Data()
    training_set = Data()
    test_set = Data()
    by_model = None
    def data_split(self, ds):

        return

    def model(self):
        clf = GaussianNB()
        result = None
        for v, l in zip(self.data_set.value, self.data_set.label):
            print(v)
            print(l)
            result = cross_validate(clf, v, l, cv=10, return_train_score=True, return_estimator=True)

            self.by_model = result['estimator']
        return result['test_score'], result['train_score']

    def display_save(self, accuracy, setname, address= '../result/'):
        for acc, bayes in zip(accuracy, self.by_model):
            print(bayes.class_prior_, bayes.class_count_, acc)
            f = open(address+setname, 'a')
            s = str(bayes.class_prior_)
            f.writelines([str(bayes.class_prior_),str(bayes.class_count_),'test_acc:  ',str(acc), '\n'])
            f.close()




t = supportvectormachine()
'''
names = ['seeds_dataset/seeds_dataset.txt']
for name in names:
    uci_data = load_data(PATH, name,'txt')
    s = uci_data.shape
    col = s[1]
    row = s[0]
    uci_label = uci_data[:,col-1]
    uci_data =uci_data[:,0:col-1]
    t.data_set.value.append(uci_data)
    t.data_set.label.append(uci_label)
    test_acc, train_acc = t.model()
    name = 'seeds_dataset/seeds_dataset_bayes.txt'
    print(t.display_save(test_acc, name))
'''
'''
names = ['CNAE-9/CNAE-9.txt']
for name in names:
    uci_data = load_data(PATH, name,'txt')
    s = uci_data.shape
    col = s[1]
    row = s[0]
    uci_label = uci_data[:,0]
    uci_data =uci_data[:,1:col-1]
    t.data_set.value.append(uci_data)
    t.data_set.label.append(uci_label)
    test_acc, train_acc = t.model()
    name = 'CNAE-9/CNAE-9_bayes.txt'
    print(t.display_save(test_acc, name))
'''
'''
names = ['diabetes-dataset/diabetes-data/Diabetes-Data/']
for name in names:
    for i in range(1,70):
        if i<10:
            uci_data = load_data(PATH, name+'data-0' + str(i) + '.txt', 'txt')
            s = uci_data.shape
            col = s[1]
            row = s[0]
            uci_label = uci_data[:, col-1]
            uci_data = uci_data[:, 0:col - 1]
            t.data_set.value.append(uci_data)
            t.data_set.label.append(uci_label)
            test_acc, train_acc = t.model()
            print(t.display_save(test_acc, name))
        else:
            uci_data = load_data(PATH, name + 'data-' + str(i) + '.txt', 'txt')

            uci_label = uci_data[:,0]
            uci_data =uci_data[:,1:col-1]
            t.data_set.value.append(uci_data)
            t.data_set.label.append(uci_label)
            test_acc, train_acc = t.model()
            print(t.display_save(test_acc, name))
'''
'''
names = ['Parkinsons/parkinsons.txt']
for name in names:
    uci_data = load_data(PATH, name,'txt')
    s = uci_data.shape
    col = s[1]
    row = s[0]
    uci_label = uci_data[:,17]
    uci_data =np.append(uci_data[:,1:17], uci_data[:,18:col],axis=1)
    t.data_set.value.append(uci_data)
    t.data_set.label.append(uci_label)
    test_acc, train_acc = t.model()
    name = 'Parkinsons/parkinsons-bayes.txt'
    print(t.display_save(test_acc, name))
'''
'''
names = ['knowledge/Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN.csv']
def trans(level):
    if level == 'very_low':
        return 0
    elif level == 'Low':
        return 1
    elif level == 'Middle':
        return 2
    else:
        return 3

for name in names:
    uci = load_data(PATH, name,'csv')
    uci_data = list()
    uci_label = list()
    for i in uci:
        a = [i['\ufeffSTG'], i['SCG'], i['STR'], i['LPR'], i['PEG']]
        b = trans(i[' UNS'])
        uci_data.append(a)
        uci_label.append(b)
    m = np.array(uci_data)
    n = np.array(uci_label)
    t.data_set.value.append(np.array(uci_data))
    t.data_set.label.append(np.array(uci_label))
    test_acc, train_acc = t.model()
    name = 'knowledge/Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN-bayes.csv'
    print(t.display_save(test_acc, name))
'''

'''
names = ['Glass/glass.txt']
for name in names:
    uci_data = load_data(PATH, name,'txt')
    s = uci_data.shape
    col = s[1]
    row = s[0]
    uci_label = uci_data[:,col-1]
    uci_data =uci_data[:,1:col-1]
    t.data_set.value.append(uci_data)
    t.data_set.label.append(uci_label)
    test_acc, train_acc = t.model()
    name = 'Glass/glass-bayes.txt'
    print(t.display_save(test_acc, name))
'''
'''
names = ['Ionosphere/ionosphere.txt']
for name in names:
    uci_data = load_data(PATH, name,'txt')
    s = uci_data.shape
    col = s[1]
    row = s[0]
    uci_label = uci_data[:,34]
    uci_data =uci_data[:,0:34]
    t.data_set.value.append(uci_data)
    t.data_set.label.append(uci_label)
    test_acc, train_acc = t.model()
    name = 'Ionosphere/ionosphere-bayes.txt'
    print(t.display_save(test_acc, name))
'''
'''
names = ['segmentation/segmentation.txt']
for name in names:
    uci_data = load_data(PATH, name,'txt')
    s = uci_data.shape
    col = s[1]
    row = s[0]
    uci_label = uci_data[:,0]
    uci_data =uci_data[:,1:34]
    t.data_set.value.append(uci_data)
    t.data_set.label.append(uci_label)
    test_acc, train_acc = t.model()
    name = 'Ionosphere/ionosphere-bayes.txt'
    print(t.display_save(test_acc, name))
    print(np.mean(test_acc))
    '''
'''
names = ['page_block/page-blocks.txt']
for name in names:
    uci_data = load_data(PATH, name,'txt')
    s = uci_data.shape
    col = s[1]
    row = s[0]
    uci_label = uci_data[:,10]
    uci_data =uci_data[:,0:10]
    t.data_set.value.append(uci_data)
    t.data_set.label.append(uci_label)
    test_acc, train_acc = t.model()
    name = 'Ionosphere/ionosphere-bayes.txt'
    print(t.display_save(test_acc, name))
    print(np.mean(test_acc))
    '''
names = ['mammographic-masses/mammographic_masses.txt']
for name in names:
    uci_data = load_data(PATH, name,'txt')
    s = uci_data.shape
    col = s[1]
    row = s[0]
    uci_label = uci_data[:,5]
    uci_data =uci_data[:,0:5]
    t.data_set.value.append(uci_data)
    t.data_set.label.append(uci_label)
    test_acc, train_acc = t.model()
    name = 'mammographic-masses/mammographic_masses-bayes.txt'
    print(t.display_save(test_acc, name))
    print(np.mean(test_acc))

