__author__ = 'jason'


from sklearn import svm
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import numpy as np
from sklearn import preprocessing


class Classifier:
    def __init__(self):
        #Data containers
        self.training_data = []
        self.training_targets = []
        # Preprocess objects
        self.le = preprocessing.LabelEncoder()
        self.scaler = preprocessing.StandardScaler()
        #Classifier Type
        self.classifier = svm.SVC()
        self.name = 'SVM'
        # self.classifier = svm.SVC(kernel='rbf'); self.name = 'SVM'
        # self.classifier = RandomForestClassifier()
        # self.name = 'Rand Forest'
        #self.classifier = AdaBoostClassifier(); self.name = 'Adaboost'
        #self.classifier = neighbors.KNeighborsClassifier(); self.name = 'KNN'

    # TODO: this is kind of bad practice
    @property
    def x(self):
        return self._x


    def add_training_data(self, obs, target, indices=0):
        if not indices:
            self.training_data.append(np.array(obs))
            self.training_targets.append(target)
        else:
            for ii in indices:
                # lol bad recursion
                self.add_training_data(obs[ii], target[ii])

    def fit(self, scale=True):
        self.training_data = np.array(self.training_data)
        self.training_targets = np.array(self.training_targets)
        if scale:
            # self.scaler.fit(np.float(self.training_data))
            self.scaler.fit(self.training_data)
            #self.training_data = self.scaler.transform(self.training_data)
            self.training_data = preprocessing.scale(self.training_data)
        self.classifier.fit(self.training_data, self.training_targets)

    def preprocess(self):
        self.training_data = preprocessing.scale(self.training_data)

    def predict(self, data, scale=True):
        data = np.array(data)
        if scale:
            #data = self.scaler.transform(data)
            data = preprocessing.scale(data)
        return self.classifier.predict(data)

