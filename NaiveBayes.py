import sys
import csv
import json

if len(sys.argv) != 3:
    print("Usage: python3 NaiveBayes.py breast-cancer-training.csv breast-cancer-test.csv")
    sys.exit()

trainingFile = sys.argv[1]
testFile = sys.argv[2]

with open(trainingFile, newline='') as csvfile:
    reader = csv.reader(csvfile)
    features = next(reader) # get the features
    instances = [row[1:] for row in reader] # get the data and skip the first column

classLabels = [row[0] for row in instances] # get the class labels
classLabels = list(dict.fromkeys(classLabels)) # remove duplicates
    
features = features[2:] # remove the empty string and class column

# hard-code the possible values for each feature (prevent 0 occurances of a value in the training set)
possibleValues = [['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99'],
                  ['lt40', 'ge40', 'premeno'],
                  ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59'],
                  ['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '18-20', '21-23', '24-26', '27-29', '30-32', '33-35', '36-39'],
                  ['yes', 'no'],
                  ['1', '2', '3'],
                  ['left', 'right'],
                  ['left_up', 'left_low', 'right_up', 'right_low', 'central'],
                  ['yes', 'no']]

print("\nSetting up =============================================\n") 
print("Possible class labels: ",classLabels)
print("Possible features",features)
print("Possible values for each feature",possibleValues)

# Initialise the count numbers to 1
classCounts = {}
counts = {}
for classLabel in classLabels:
    classCounts[classLabel] = 1
    counts[classLabel] = {}
    for feature in features:
        counts[classLabel][feature] = {}
        for value in possibleValues[features.index(feature)]:
            counts[classLabel][feature][value] = 1

print("\nCounts after initialising to 1 ==========================\n")  
print("Class counts: ", json.dumps(classCounts, indent = 2))
print("Feature counts: ", json.dumps(counts, indent = 2))

# Count the number of occurances of each value for each feature for each class
for instance in instances:
    classCounts[instance[0]] += 1
    for feature in features:
        value = instance[features.index(feature)]
        counts[instance[0]][feature][instance[features.index(feature) + 1]] += 1

print("\nCounts after counting ===================================\n")  
print("Class counts: ",classCounts)
print("Feature counts: ", json.dumps(counts, indent = 2))

# Calculate the total/denominators
classTotal = 0
countTotal = {}
for classLabel in classLabels:
    classTotal += classCounts[classLabel]
    countTotal[classLabel] = {}
    for feature in features:
        countTotal[classLabel][feature] = 0
        for value in possibleValues[features.index(feature)]:
            countTotal[classLabel][feature] += counts[classLabel][feature][value]

print("\nTotals ==================================================\n")
print("Class total: ",classTotal)
print("Feature totals: ", json.dumps(countTotal, indent = 2))

# Calculate the probabilities from the counting numbers
classProbabilities = {}
probabilities = {}
for classLabel in classLabels:
    # store a tuple of the fraction and the decimal value
    classProbabilities[classLabel] = (str(classCounts[classLabel])+"/"+str(classTotal),classCounts[classLabel] / classTotal)
    probabilities[classLabel] = {}
    for feature in features:
        probabilities[classLabel][feature] = {}
        for value in possibleValues[features.index(feature)]:
            # store a tuple of the fraction and the decimal value
            probabilities[classLabel][feature][value] = \
                (str(counts[classLabel][feature][value])+"/"+str(countTotal[classLabel][feature]),\
                 counts[classLabel][feature][value] / countTotal[classLabel][feature])

print("\nProbabilities from training ==============================\n")
print("Class probabilities: ",json.dumps(classProbabilities, indent = 2))
print("Feature probabilities: ",json.dumps(probabilities, indent = 2))

# Read the test file
with open(testFile, newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # skip the first row
    testInstances = [row[1:] for row in reader] # get the data and skip the first column

# Calculation of the class scores for each instance, and the predicted class
print("\nTest instances ===========================================\n")
score1 = classProbabilities[classLabels[0]][1]
score2 = classProbabilities[classLabels[1]][1] 
print("P(Y = ",classLabels[0],") = ",score1)
print("P(Y = ",classLabels[1],") = ",score2)
print("\n")

predictions = []
for instance in testInstances:
    score1 = classProbabilities[classLabels[0]][1]
    score2 = classProbabilities[classLabels[1]][1] 
    for feature in features:
        score1 *= probabilities[classLabels[0]][feature][instance[features.index(feature)+1]][1]
        score2 *= probabilities[classLabels[1]][feature][instance[features.index(feature)+1]][1]

    print("P(Y = ",classLabels[0],", X =",instance[1:],") = ",score1)
    print("P(Y = ",classLabels[1],", X =",instance[1:],") = ",score2)
    prediction = classLabels[0] if score1 > score2 else classLabels[1]
    print("predicted class of the input vector: ", prediction)
    predictions.append(prediction)
    print("\n")

# Calculate the accuracy
count = 0
for i in range(len(testInstances)):
    if testInstances[i][0] == predictions[i]:
        count += 1
accuracy = count / len(testInstances)
print("Accuracy: ",accuracy)


