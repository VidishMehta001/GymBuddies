import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import pickle

classes = ['No Exercise','squats','pushups']

data = np.load('./Video Saving/hog_data.npy')
labels = np.load('./Video Saving/hog_label.npy')

x, y, z = data.shape
data = np.reshape(data, (x, y))
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=42, shuffle = True)

#clf = svm.LinearSVC(C=100, penalty='l2', loss='hinge', random_state=42, verbose=0, max_iter=1000)
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

with open('hog_svm_model.pkl', 'wb') as file:
    pickle.dump(clf, file)

y_pred = clf.predict(X_test)
print(y_pred, y_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test,
                                    y_pred,
                                    target_names=classes,
                                    digits=4))
confusion   = metrics.confusion_matrix(y_test,y_pred)
print(confusion)

