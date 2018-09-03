import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, mean_squared_error,mean_absolute_error, roc_curve, classification_report,auc)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)

def label_encode(x, cols):
	for i in cols:
		labelencoder = LabelEncoder()
		x[:, i] = labelencoder.fit_transform(x[:, i])

def one_hot_encode(x, cols):
	for i in cols:
		onehotencoder = OneHotEncoder(categorical_features = [i])
		x = onehotencoder.fit_transform(x).toarray()

def model_training(model, x_train, x_test, y_train, y_test):
	print("***************************************************************")
	print(model.clf)
	print("model training...")
	t0 = time.time();
	model.fit(x_train, y_train)#在训练集训练模型
	train_time = time.time() - t0

	t0 = time.time();
	expected = y_test
	predicted = model.predict(x_test)#在测试集进行测试
	test_time = time.time() - t0

	accuracy = accuracy_score(expected, predicted)
	recall = recall_score(expected, predicted, average="binary")
	precision = precision_score(expected, predicted , average="binary")
	f1 = f1_score(expected, predicted , average="binary")
	cm = confusion_matrix(expected, predicted)#混淆矩阵
	tpr = float(cm[0][0])/np.sum(cm[0])
	fpr = float(cm[1][1])/np.sum(cm[1])

	print(cm)
	print("tpr:%.3f" %tpr)
	print("fpr:%.3f" %fpr)
	print("accuracy:%.3f" %accuracy)
	print("precision:%.3f" %precision)
	print("recall:%.3f" %recall)
	print("f-score:%.3f" %f1)
	print("train_time:%.3fs" %train_time)
	print("test_time:%.3fs" %test_time)
	print("***************************************************************")

print("load data...")
dataset = pd.read_csv('kddcup.data_10_percent_corrected')#此数据集越有49万条数据，共42列，最后一列为类别标签

print("change Multi-class to binary-class...")
dataset['normal.'] = dataset['normal.'].replace(['back.', 'buffer_overflow.', 'ftp_write.', 'guess_passwd.', 'imap.', 'ipsweep.', 'land.', 'loadmodule.', 'multihop.', 'neptune.', 'nmap.', 'perl.', 'phf.', 'pod.', 'portsweep.', 'rootkit.', 'satan.', 'smurf.', 'spy.', 'teardrop.', 'warezclient.', 'warezmaster.'], 'attack')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 41].values

print("encoding categorical data...")
cols1 = [1, 2, 3]
cols2 = [1, 4, 70]
label_encode(x, cols1)
one_hot_encode(x, cols2)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

print("splitting the dataset...")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

#feature scaling
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }

# Create 5 objects that represent our 4 models
SEED = 0
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

model_training(rf, x_train, x_test, y_train, y_test)
model_training(et, x_train, x_test, y_train, y_test)
model_training(ada, x_train, x_test, y_train, y_test)
model_training(gb, x_train, x_test, y_train, y_test)
model_training(svc, x_train, x_test, y_train, y_test)
