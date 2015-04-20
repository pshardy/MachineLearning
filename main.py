__author__ = 'Patrick Hardy'

import hashlib
import time
import pandas as pd
from sklearn import cross_validation

# Hash a string into a unique int.
def hash_str(s):
    return int(hashlib.sha256(s).hexdigest(), 16)


def fit_and_score(clf, x_train, x_test, y_train, y_test):
    # Profile time to fit.
    start = time.clock()
    print "Fitting with " + str(clf.fit(x_train, y_train))
    end = time.clock()

    # Profile prediction time.
    start_predict = time.clock()
    result = clf.score(x_test, y_test) #clf.predict(test)
    end_predict = time.clock()

    print "\nAccuracy: %0.2f%% (+/- %0.2f)" % (result.mean() * 100.0, result.std() * 2 * 100)

    print "\nTime to fit in seconds: %0.5f" % (end - start)
    print "Time to predict in seconds: %0.5f\n" % (end_predict - start_predict)

    return result


#myData = np.genfromtxt("data_files\\Book1.csv", dtype=float, delimiter=',', skip_header=0)

# Read the training data from csv.
#crx.data.csv
data = pd.read_csv("data_files\\crx.data.csv", header=None, engine='c', na_values=['?'], true_values=['+'],
                   false_values=['-'], error_bad_lines=True,
                   dtype={0: 'object', 1: 'float32', 2: 'float32', 3: 'object', 4: 'object', 5: 'object', 6: 'object',
                          7: 'float32', 8: 'object', 9: 'object', 10: 'float32', 11: 'object', 12: 'object',
                          13: 'float32', 14: 'float32', 15: 'object'},
                   converters={0: hash_str, 3: hash_str, 4: hash_str, 5: hash_str, 6: hash_str,
                               8: hash_str, 9: hash_str, 11: hash_str, 12: hash_str})


# Remove ? values all together.
data = data.dropna()

# Choose features.
x = data.ix[:, 0:14]
# Choose results.
y = data.ix[:, 15]

# Cross-validation data.
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2, random_state=0)

print 'Samples, Features ' + str(data.shape)

# TODO: Change this to test different classifiers.
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

# Train with the sample data.
kn_result = fit_and_score(KNeighborsClassifier(algorithm='kd_tree', n_neighbors=5), x_train, x_test, y_train, y_test)
svc_result = fit_and_score(LinearSVC(dual=False), x_train, x_test, y_train, y_test)

print "Method 1 performs with %0.3f accuracy difference" % ((kn_result - svc_result) * 100.0)

#print result

# Test the original values to see if the same results are produced, not really useful and can lean to over-fitting.
#test = x.ix[:, 0:14]
# Check for match percent.
#success_count = 0
#for index, item in enumerate(result.data):
#    if index < len(yArr) and item == yArr[index]:
#        success_count += 1

#print 'Accuracy: ' + str((float(success_count) / float(len(yArr))) * 100.0)