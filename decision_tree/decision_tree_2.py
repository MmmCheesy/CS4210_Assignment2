#-------------------------------------------------------------------------
# AUTHOR: Alexander Eckert
# FILENAME: decision_tree_2.py
# SPECIFICATION: Training 3 models with different data and comparing accuracy.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1.2 hrs
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    # Mapping for feature transformation
    age_mapping = {'Young': 1, 'Presbyopic': 3, 'Prepresbyopic': 2}
    prescription_mapping = {'Myope': 1, 'Hypermetrope': 2}
    astigmatism_mapping = {'Yes': 1, 'No': 2}
    tear_production_mapping = {'Reduced': 1, 'Normal': 2}

    # Mapping for class transformation
    lenses_mapping = {'Yes': 1, 'No': 2}

    for row in dbTraining:
        # Transforming features
        features = [
            age_mapping[row[0]],
            prescription_mapping[row[1]],
            astigmatism_mapping[row[2]],
            tear_production_mapping[row[3]]
        ]
        X.append(features)

        # Transforming classes
        Y.append(lenses_mapping[row[4]])

    #loop your training and test tasks 10 times here
    total_accuracy = 0
    for i in range (10):

        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)

        #reading the test data in a csv file
        dbTest = []
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0: #skipping the header
                    dbTest.append (row)

        X2 = []
        Y2 = []
        for data in dbTest:
            # Transforming features for the test data
            features_test = [
                age_mapping[data[0]],
                prescription_mapping[data[1]],
                astigmatism_mapping[data[2]],
                tear_production_mapping[data[3]]
            ]
            X2.append(features_test)
            Y2.append(lenses_mapping[data[4]])

        # Predict using the trained decision tree
        predictions = clf.predict(X2)

        # Calculate accuracy for this run
        correct_predictions = sum(predictions == Y2)
        accuracy = correct_predictions / len(Y2)

        # Accumulate accuracy for averaging later
        total_accuracy += accuracy

    # Calculate average accuracy over 10 runs
    average_accuracy = total_accuracy / 10

    # Print the average accuracy
    print(f'Final accuracy when training on {ds}: {average_accuracy}')




