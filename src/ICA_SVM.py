# Import matplotlib library
import matplotlib.pyplot as plt
  
# Import scikit-learn library
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix
from sklearn import manifold
from sklearn.decomposition import FastICA
from time import time
from sklearn import metrics
import seaborn as sns

import numpy as np

# download the LFW_people's dataset
lfw_people = fetch_lfw_people(min_faces_per_person = 70, resize = 0.4)
  
# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape
  
# Instead of providing 2D data, X has data already in the form  of a vector that
# is required in this approach.
X = lfw_people.data
n_features = X.shape[1]
  
# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]
  
# Print Details about dataset
print("Number of Data Samples: % d" % n_samples)
print("Size of a data sample: % d" % n_features)
print("Number of Class Labels: % d" % n_classes)

# Visualization
def plot_gallery(images, titles, h, w, rows=3, cols=4):
    plt.figure()
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())

def titles(y_pred, y_test, target_names):
    for i in range(y_pred.shape[0]):
        pred_name = target_names[y_pred[i]].split(' ')[-1]
        true_name = target_names[y_test[i]].split(' ')[-1]
        yield 'predicted: {0}\ntrue: {1}'.format(pred_name, true_name)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.25, random_state = 42)
print("size of training Data is % d and Testing Data is % d" %(
        y_train.shape[0], y_test.shape[0]))
        
n_components = 200

t0 = time()

ica = FastICA(n_components=n_components,
          whiten = True).fit(X_train)

eigenfaces = ica.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")

X_train_ica = ica.transform(X_train)
X_test_ica = ica.transform(X_test)

#
print("Sample Data point after applying ICA\n", X_train_ica[0])
print("-----------------------------------------------------")
print("Dimesnsions of training set = % s and Test Set = % s"%(
        X_train.shape, X_test.shape))
        
print("Fitting the classifier to the training set")

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(
    SVC(kernel ='rbf', class_weight ='balanced'), param_grid
)
clf = clf.fit(X_train_ica, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)
  
print("Predicting people's names on the test set")

y_pred = clf.predict(X_test_ica)

print("Accuracy score:{:.2f}%".format(metrics.accuracy_score(y_test, y_pred)*100))

plt.figure(2, figsize=(16, 9))
sns.heatmap(metrics.confusion_matrix(y_test, y_pred))

# print classifiction results
print(classification_report(y_test, y_pred, target_names = target_names))
# print confusion matrix
print("Confusion Matrix is:")
print(confusion_matrix(y_test, y_pred, labels = range(n_classes)))

# plot confusion matrix
plt.rcParams.update({'font.size': 10})
plt.rcParams['figure.figsize'] = [10, 10]
plot_confusion_matrix(clf, X_test_ica, y_test, cmap=plt.cm.Blues, display_labels=range(n_classes))

prediction_titles = list(titles(y_pred, y_test, target_names))
plot_gallery(X_test, prediction_titles, h, w)

eigenface_titles = ["eigenface {0}".format(i) for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()
