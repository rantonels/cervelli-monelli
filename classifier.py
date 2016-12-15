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


from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

labels_names = [ "gender", "age", "health" ]

def load_pickle_dataset(fname):
    
    db = np.load(fname)

    return db

    out_array = np.zeros((db.shape[0],db.shape[1]))

    for i in range(len(db)):
        out_array[i,:] = db[i+1].reshape(-1)

    return out_array


def blist2str(l):
    return "".join(map(str,l))

def train(): # addestra i classificatori e stampa info

    targets = {}
    with open("data/mlp3-targets.csv",'rb') as csvfile:
        reader = csv.reader(csvfile,delimiter = ",")
        index = 1
        for row in reader:
            entry = map( int, row)
            targets[index] = entry
            index+=1


    samples_array = load_pickle_dataset("features/train_dataset.npy")





   # total_number = samples_array.shape[0]
   # train_number = 230 # number of entries from train_dataset used for training
   # test_number = total_number - train_number


    hamming_loss = {}
    
    classifiers = {}

    for label in range(3): 
        print()
        print("* Classifico labella %s"%labels_names[label])


        # Create targets array
        targets_array = np.zeros((len(targets)))
        for i in range(len(targets)):
            targets_array[i] = targets[i+1][label]


        # feature selection


        if False:
            print("feature selection...")
            clf = ExtraTreesClassifier()

            clf = clf.fit(samples_array, targets_array)

            model = SelectFromModel(clf, prefit=True)
            

            samples_extracted = model.transform(samples_array)

            print("Feature ridotte %d -> %d"% (samples_array.shape[1], samples_extracted.shape[1]))
        else:
            samples_extracted = samples_array



        print("split..")

        # split

        samples_train, samples_test, targets_train, targets_test = train_test_split( 
                samples_extracted, 
                targets_array, 
                test_size=0.2, 
                random_state=np.random.randint(0,1000)
                )


  
        print("init..")

        # init classifier

        if label == 1: # eta'
            #classifier = GaussianProcessClassifier(1.0*RBF(1.0), warm_start=True,n_jobs = 9)
            #classifier = svm.SVC(gamma=1.0, C = 1.0, class_weight = "balanced"
            classifier = svm.SVC(kernel="linear")
            #classifier = GaussianProcessClassifier(2.5*RBF(2.5), warm_start=True,n_jobs = 9,max_iter_predict=1000) # meh, non male for age
            classifier = DecisionTreeClassifier(max_depth=50, criterion="entropy") # non ci siamo
        elif label == 2: # salute
            classifier = svm.SVC(kernel="linear",max_iter=1000000)
            #classifier = svm.SVC(gamma=1.0, C = 1.0, class_weight = "balanced")
            #classifier = GaussianProcessClassifier(1.0*RBF(1.0), warm_start=True,n_jobs = 9)
        else: # gender
            # questi sono i migliori fin'ora
            
            #classifier = svm.SVC(gamma='auto', C = 1.0, class_weight = "balanced")
            #classifier = GaussianProcessClassifier(1.0*RBF(1.0), warm_start=True,n_jobs = 9,max_iter_predict=1000)
            classifier = svm.SVC(kernel="linear",max_iter=1000000)
            #classifier = DecisionTreeClassifier(max_depth=10, criterion="entropy")

            # questi altri signori, schifo

            #classifier = svm.SVC(gamma='auto', C = 100, class_weight = "balanced")
            #classifier = GaussianProcessClassifier(4.0*ConstantKernel(), warm_start=True,n_jobs = 9,max_iter_predict=1000)
            #classifier = MLPClassifier(alpha=2.0)
            #classifier = KNeighborsClassifier(10)
        
            #classifier = AdaBoostClassifier(n_estimators = 500) # good for age
        #classifier = MLPClassifier(alpha=3.0,activation="tanh",tol=1e-6) # a volte good for age


        #classifier = svm.SVC(kernel="linear", C= 1.0) # nah
        

        print("fit..")

        classifier.fit(samples_train,targets_train)

        expected_output = targets_test.astype(int)
        predicted_output = classifier.predict(samples_test).astype(int)
    

        print("\tOutput atteso:  \t",blist2str(expected_output[0:50]),"...")
        print("\tOutput predetto:\t",blist2str(predicted_output[0:50]),"...")

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


exit()

test_array = load_pickle_dataset("features/test_dataset")

test_output = {}

output_strink = "ID\tSample\tLabel\tPredicted\n"

for label in range(3):
    test_output[label] = good_classifiers[label].predict(test_array).astype(int)

num_data = test_array.shape[0]


with open("submission.csv",'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow([
        "ID", "Sample", "Label", "Predicted"
    ])
    for i in range(num_data):
        for label in range(3):
            writer.writerow([
                3 * i + label,
                i,
                labels_names[label],
                bool(test_output[label][i]) 
            ])
