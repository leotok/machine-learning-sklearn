# Machine Learning tutorial
# http://machinelearningmastery.com/machine-learning-in-python-step-by-step/

import pandas
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Load dataset

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = [
    'sepal-length', 'sepal-width',
    'petal-length','petal-width',
    'class'
    ]

dataset = pandas.read_csv(url, names=names)

# shape
# print dataset.shape

# # head
# print dataset.head(20)

# # descriptions
# print dataset.describe()

# # class distribution
# print dataset.groupby('class').size()

# ########## Data Visualization ###########

# # box and whisker plots
# dataset.plot(
#     kind='box', subplots=True,
#     layout=(2,2), sharex=False,
#     sharey=False
#     )
# plt.show()

# # histograms
# dataset.hist()
# plt.show()

# # scatter plot matrix
# scatter_matrix(dataset)
# plt.show()

######### Validation Dataset ##########

# split-out Validation dataset
array = dataset.values
# assign all rows from data columns to X
X = array[:,0:4]
# assign all rows from class column to Y
Y = array[:,4]
validation_size = 0.20
seed = 7
# separate data into test and validation arrays
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(
    X, Y, test_size=validation_size, random_state=seed
)

print "X_train", len(X_train), "X_validation", len(X_validation), "Y_train", len(Y_train), "Y_validation", len(Y_validation)

###### Test Harness ######

# test options and evaluation metrics
num_folds = 10
num_instances = len(X_train)
scoring = 'accuracy'

# Spot Check Algorithims
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []

for name, model in models:
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print msg

# Compare Algorithims
# fig = plt.figure()
# fig.suptitle('Algorithims Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()
