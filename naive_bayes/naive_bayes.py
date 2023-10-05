#-------------------------------------------------------------------------
# AUTHOR: Alexander Eckert
# FILENAME: naive_bayes.py
# SPECIFICATION: Naive Bayes
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1.5 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#reading the training data in a csv file
db = []
X = []
Y = []
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #sipping the header
            db.append(row)

outlook = {'Sunny': 1, 'Overcast': 2, 'Rain': 3}
temperature = {'Hot': 1, 'Cool': 2, 'Mild': 3}
humidity = {'Normal': 1, 'High': 2}
wind = {'Strong': 1, 'Weak': 2}
play = {'Yes': 1, 'No': 2}

for item in db:
    X.append([outlook[item[1]], temperature[item[2]], humidity[item[3]], wind[item[4]]])
    Y.append(play[item[5]])

clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
test_data = []
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            test_data.append(row)

print('Day    Outlook     Temperature    Humidity    Wind    Play Tennis     Confidence')

#use your test samples to make probabilistic predictions.
for i, test_sample in enumerate(test_data, start=15):  # starting from Day 15 as per the desired format
    test_sample = [outlook[test_sample[1]], temperature[test_sample[2]], humidity[test_sample[3]], wind[test_sample[4]]]
    predictions = clf.predict_proba([test_sample])[0]
    day = f'D{i}'  # generate the day string
    outlook_val = [k for k, v in outlook.items() if v == test_sample[0]][0]
    temperature_val = [k for k, v in temperature.items() if v == test_sample[1]][0]
    humidity_val = [k for k, v in humidity.items() if v == test_sample[2]][0]
    wind_val = [k for k, v in wind.items() if v == test_sample[3]][0]
    play_val = 'Yes' if predictions[0] > predictions[1] else 'No'
    confidence = max(predictions[0], predictions[1])
    print(f'{day}    {outlook_val}    {temperature_val}    {humidity_val}    {wind_val}    {play_val}    {confidence:.5f}')

