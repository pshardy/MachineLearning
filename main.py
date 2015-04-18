__author__ = 'Patrick Hardy'

import hashlib
import time
import pandas as pd


# Hash a string into a unique int.
def hash_str(s):
    return int(hashlib.sha256(s).hexdigest(), 16)

#myData = np.genfromtxt("data_files\\crx.data.csv", dtype=float, delimiter=',', skip_header=0)

# Read the training data from csv.
data = pd.read_csv("data_files\\crx.data.csv", header=None, engine='c', na_values=['?'], true_values=['+'],
                   false_values=['-'], error_bad_lines=True,
                   dtype={0: 'object', 1: 'float32', 2: 'float32', 3: 'object', 4: 'object', 5: 'object', 6: 'object',
                          7: 'float32', 8: 'object', 9: 'object', 10: 'float32', 11: 'object', 12: 'object',
                          13: 'float32', 14: 'float32', 15: 'object'},
                   converters={0: hash_str, 3: hash_str, 4: hash_str, 5: hash_str, 6: hash_str,
                               8: hash_str, 9: hash_str, 11: hash_str, 12: hash_str})


# Remove ? values all together.
data = data.dropna()

x = data.ix[:, 0:14]
y = data.ix[:, 15]

# Format target as single array.
yArr = []

for i in y:
    yArr.append(i)

print 'Samples, Features ' + str(data.shape)

# TODO: Change this to test different classifiers.
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

# Train with the sample data.
clf = KNeighborsClassifier()

# Profile time to fit.
start = time.clock()
print "Fitting with " + str(clf.fit(x, yArr))
end = time.clock()

print "\nTime to fit in seconds: " + str(end - start)

# Test the original values to see if the same results are produced.
test = x.ix[:, 0:14]

# Profile prediction time.
start = time.clock()
result = clf.predict(test)
end = time.clock()

print "Time to predict in seconds: " + str(end - start)

#print result

# Check for match percent.
success_count = 0
for index, item in enumerate(result.data):
    if index < len(yArr) and item == yArr[index]:
        success_count += 1

print 'Accuracy: ' + str((float(success_count) / float(len(yArr))) * 100.0)