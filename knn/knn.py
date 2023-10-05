#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []
classes = {"-": 1, "+": 2} #convert classes to numbers
e = 0 #error

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            db.append (row)

#loop your data to allow each instance to be your test set
for test_instance in db:
    i = db.index(test_instance)

    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration and add the vector Y.
    X = []
    Y = []
    for j, row in enumerate(db):
        if j != i:  # Exclude the current test instance
            X.append([float(val) for val in row[:-1]])  # Convert features to float
            Y.append(classes[row[-1]])  # Map class label to its numerical representation

    #store the test sample of this iteration in the vector testSample
    testSample = [float(val) for val in test_instance[:-1]]

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction.
    class_predicted = clf.predict([testSample])[0]

    #compare the prediction with the true label of the test instance to start calculating error rate
    true_label = classes[test_instance[-1]]
    if class_predicted != true_label:
        e += 1  # Increment the error count

#print the error rate
error_rate = e / len(db)
print("Error rate:", error_rate)
