from __future__ import print_function

import pickle
import csv
import numpy as np
from sklearn import svm,metrics
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

labels_names = [ "GENDER", "age", "health" ]

def load_pickle_dataset(fname):
    
    db = pickle.load(open(fname,'r'))

    out_array = np.zeros((len(db),db[1].size))

    for i in range(len(db)):
        out_array[i,:] = db[i+1].reshape(-1)

    return out_array


def train():

    targets = {}
    with open("targets.csv",'rb') as csvfile:
        reader = csv.reader(csvfile,delimiter = ",")
        index = 1
        for row in reader:
            entry = map( int, row)
            targets[index] = entry
            index+=1


    samples_array = load_pickle_dataset("features/train_dataset")


    total_number = samples_array.shape[0]
    train_number = 230 # number of entries from train_dataset used for training
    test_number = total_number - train_number



    
    classifiers = {}

    for label in range(3):
        
        targets_array = np.zeros((len(targets)))
        for i in range(total_number):
            targets_array[i] = targets[i+1][label]

        print("Classifico labella %s"%labels_names[label])


        #classifier = svm.SVC(gamma='auto', C = 100.0, class_weight = "balanced")
        classifier = GaussianProcessClassifier(1.0*RBF(1.0), warm_start=True,n_jobs = 9)
        #classifier = MLPClassifier(alpha=0.5,activation="tanh",tol=1e-6)
        #classifier = AdaBoostClassifier(n_estimators = 1000)

        classifier.fit(samples_array[:train_number],targets_array[:train_number])

        expected_output = targets_array[train_number:]
        predicted_output = classifier.predict(samples_array[train_number:])
    

        print("Output atteso:",expected_output)
        print("Output predetto:",predicted_output)

        print("Report classificazione:")
        print(metrics.classification_report(expected_output,predicted_output))

        print("Matrice di confusione:")
        print(metrics.confusion_matrix(expected_output,predicted_output))

        classifiers[label] = classifier

    return classifiers



good_classifiers = train()


