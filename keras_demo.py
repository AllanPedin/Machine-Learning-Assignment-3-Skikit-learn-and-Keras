# Keras for deep neural networks
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import sklearn as sklearn
import sklearn.metrics
import sklearn.preprocessing
import numpy as np
import scipy

#get p-values to determine statistical significance
def get_p_value(distribution_a, distribution_b):
    #use statistical significance tests to determine which classifier performed the best
    #important -- we are using a paired (dependent) t-test. This is because the samples are
    # the same in both classifiers. Paired t-test is stronger statistically
    # We are using two-sided t-test because we want to know if the systems are different
    # (better or worse)
    t_test_result = scipy.stats.ttest_rel(distribution_a, distribution_b)
    return t_test_result.pvalue

# load the iris data
file_path = 'myData.data'
data = np.loadtxt(file_path, delimiter=',')
data_cols = np.array([False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True])
#^thats gross but it gets the job done
X = data[:,data_cols]
Y = data[:,1]


# get a test set
num_samples, num_features = X.shape
train_cutoff = int(num_samples * 0.8)
X, Y = sklearn.utils.shuffle(X, Y)
X_train = X[:train_cutoff, :]
Y_train = Y[:train_cutoff]
X_test = X[train_cutoff:, :]
Y_test = Y[train_cutoff:]

# neural networks are sensitive to scale, so normalize the data
sc = sklearn.preprocessing.StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# convert Y_train into a 3-D vector as expected by Keras
num_samples, num_features = X_train.shape
num_classes = int(max(Y_train) + 1)
Y_new = np.zeros([num_samples, num_classes])
for i in range(0, num_samples):
    Y_new[i, int(Y_train[i])] = 1
Y_train = Y_new

# construct a multi-class neural network
# model5 = Sequential()
# model5.add(Dense(33, input_shape=(num_features,), activation='tanh'))
# model5.add(Dense(33, activation='tanh'))
# model5.add(Dense(33, activation='tanh'))
# model5.add(Dense(33, activation='tanh'))
# model5.add(Dense(num_classes, activation='softmax'))
# #model.compile(Adam(lr=0.04), 'bce', metrics=['accuracy'])
# model5.compile(Adam(lr=0.04), 'categorical_crossentropy', metrics=['accuracy'])
# # train the neural network
# history5 = model5.fit(X_train, Y_train, epochs=100,verbose=0)  # , batch_size=100)

# # evaluate the model on the test data
# y_pred5 = model5.predict(X_test)
# predictions5 = np.argmax(y_pred5, axis=1)
# f15 = sklearn.metrics.f1_score(Y_test, predictions5, average='macro')
# print("f1-score = ", f15)


# # construct a multi-class neural network
# model10 = Sequential()
# model10.add(Dense(33, input_shape=(num_features,), activation='tanh'))
# model10.add(Dense(33, activation='tanh'))
# model10.add(Dense(33, activation='tanh'))
# model10.add(Dense(33, activation='tanh'))
# model10.add(Dense(33, activation='tanh'))
# model10.add(Dense(33, activation='tanh'))
# model10.add(Dense(33, activation='tanh'))
# model10.add(Dense(33, activation='tanh'))
# model10.add(Dense(33, activation='tanh'))
# model10.add(Dense(num_classes, activation='softmax'))
# #model.compile(Adam(lr=0.04), 'bce', metrics=['accuracy'])
# model10.compile(Adam(lr=0.04), 'categorical_crossentropy', metrics=['accuracy'])
# # train the neural network
# history10 = model10.fit(X_train, Y_train, epochs=100,verbose=0)  # , batch_size=100)

# # evaluate the model on the test data
# y_pred10 = model10.predict(X_test)
# predictions10 = np.argmax(y_pred10, axis=1)
# f110 = sklearn.metrics.f1_score(Y_test, predictions10, average='macro')
# print("f1-score = ", f110)

# model3 = Sequential()
# model3.add(Dense(33, input_shape=(num_features,), activation='tanh'))
# model3.add(Dense(33, activation='tanh'))
# model3.add(Dense(num_classes, activation='softmax'))
# #model.compile(Adam(lr=0.04), 'bce', metrics=['accuracy'])
# model3.compile(Adam(lr=0.04), 'categorical_crossentropy', metrics=['accuracy'])
# # train the neural network
# history3 = model3.fit(X_train, Y_train, epochs=100,verbose=0)  # , batch_size=100)

# # evaluate the model on the test data
# y_pred3 = model3.predict(X_test)
# predictions3 = np.argmax(y_pred3, axis=1)
# f13 = sklearn.metrics.f1_score(Y_test, predictions3, average='macro')
# print("f1-score = ", f13)


modelC = Sequential()
modelC.add(Dense(33, input_shape=(num_features,), activation='tanh'))
modelC.add(Dense(33, activation='relu'))
modelC.add(Dense(33, activation='selu'))
modelC.add(Dense(33, activation='sigmoid'))
modelC.add(Dense(num_classes, activation='softmax'))
#model.compile(Adam(lr=0.04), 'bce', metrics=['accuracy'])
modelC.compile(Adam(lr=0.04), 'categorical_crossentropy', metrics=['accuracy'])
# train the neural network
historyC = modelC.fit(X_train, Y_train, epochs=100,verbose=0)  # , batch_size=100)

# evaluate the model on the test data
y_predC = modelC.predict(X_test)
predictionsC = np.argmax(y_predC, axis=1)
f1C = sklearn.metrics.f1_score(Y_test, predictionsC, average='macro')
print("f1-score = ", f1C)



# Calcualte performance metrics
# f1 = sklearn.metrics.f1_score(Y_test, predictions, average='macro')
# accuracy = sklearn.metrics.accuracy_score(Y_test, predictions)
# print("f1-score = ", f1)
# print("accuracy = ", accuracy)

# Summarize the results
# print("Performance Summary:")
# print(sklearn.metrics.classification_report(Y_test, predictions))
# print("Confusion Matrix:")
# print(sklearn.metrics.confusion_matrix(Y_test, predictions))


print("5,10", get_p_value(predictions5, predictions10))
print("5,3", get_p_value(predictions5, predictions3))
print("5,C", get_p_value(predictions5, predictionsC))

print("10,3", get_p_value(predictions10, predictions3))
print("10,C", get_p_value(predictions10, predictionsC))

print("3,C", get_p_value(predictions3, predictionsC))