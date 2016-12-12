from __future__ import print_function

import pickle
import csv
import numpy as np
from sklearn import svm,metrics
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF,ConstantKernel
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

labels_names = [ "gender", "age", "health" ]

def load_pickle_dataset(fname):
    
    db = pickle.load(open(fname,'r'))

    out_array = np.zeros((len(db),db[1].size))

    for i in range(len(db)):
        out_array[i,:] = db[i+1].reshape(-1)

    return out_array


def blist2str(l):
    return "".join(map(str,l))

def train(): # addestra i classificatori e stampa info

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


    hamming_loss = {}
    
    classifiers = {}

    for label in range(3):
        
        targets_array = np.zeros((len(targets)))
        for i in range(total_number):
            targets_array[i] = targets[i+1][label]

        print()
        print("* Classifico labella %s"%labels_names[label])


        if label in [1,2]: # non gender, questo se la cava egregiamente
            classifier = GaussianProcessClassifier(1.0*RBF(1.0), warm_start=True,n_jobs = 9)

        else: # gender

            # questi sono i migliori fin'ora
            
            #classifier = GaussianProcessClassifier(2.5*RBF(2.5), warm_start=True,n_jobs = 9,max_iter_predict=1000)
            classifier = DecisionTreeClassifier(max_depth=2, criterion="entropy")

            # questi altri signori, schifo

            #classifier = svm.SVC(gamma=2.0, C = 2.0, class_weight = "balanced")
            #classifier = svm.SVC(kernel="linear", C=0.025)
            #classifier = GaussianProcessClassifier(4.0*ConstantKernel(), warm_start=True,n_jobs = 9,max_iter_predict=1000)
            #classifier = MLPClassifier(alpha=100,activation="tanh",tol=1e-6)
            #classifier = KNeighborsClassifier(10)
            #classifier = AdaBoostClassifier(n_estimators = 5000)

        classifier.fit(samples_array[:train_number],targets_array[:train_number])

        expected_output = targets_array[train_number:].astype(int)
        predicted_output = classifier.predict(samples_array[train_number:]).astype(int)
    

        print("\tOutput atteso:  \t",blist2str(expected_output))
        print("\tOutput predetto:\t",blist2str(predicted_output))

        print("\t",metrics.classification_report(expected_output,predicted_output))

        print(np.array_str(metrics.confusion_matrix(expected_output,predicted_output),max_line_width=100))

        classifiers[label] = classifier

        hamming_loss[label] = 1.0/float(expected_output.shape[0]) * np.sum( np.logical_xor(expected_output,predicted_output))

        print("\tHamming loss: %03f"%hamming_loss[label])
        
        print()

    print("---OVERALL---")
    total_hamming_loss = sum(hamming_loss.values())/3.0

    print("Total Hamming loss: %03f"%total_hamming_loss)
    print()

    return classifiers



good_classifiers = train()


test_array = load_pickle_dataset("features/test_dataset")

test_output = {}

output_strink = "ID\tSample\tLabel\tPredicted\n"

for label in range(3):
    test_output[label] = good_classifiers[label].predict(test_array).astype(int)

num_data = test_array.shape[0]

bool_words = { 0: "FALSE", 1: "TRUE" }

for i in range(num_data):
    for label in range(3):
        output_strink +=    str(3*i + label) + "\t" \
                            + str(i) + "\t" \
                            + labels_names[label] + "\t"\
                            + bool_words[ test_output[label][i] ] + "\n"

f = open("submission",'w')
f.write(output_strink)
f.close()
