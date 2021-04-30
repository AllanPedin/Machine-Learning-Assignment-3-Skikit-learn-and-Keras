# Keras for deep neural networks
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import sklearn as sklearn
import sklearn.metrics
import sklearn.preprocessing
import numpy as np

# load the iris data
file_path = 'iris.data'
data = np.loadtxt(file_path, delimiter=',')
X = data[:, :-1]
Y = data[:, -1]


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
model = Sequential()
model.add(Dense(10, input_shape=(num_features,), activation='tanh'))
model.add(Dense(8, activation='tanh'))
model.add(Dense(6, activation='tanh'))
model.add(Dense(num_classes, activation='softmax'))
#model.compile(Adam(lr=0.04), 'bce', metrics=['accuracy'])
model.compile(Adam(lr=0.04), 'categorical_crossentropy', metrics=['accuracy'])
# train the neural network
history = model.fit(X_train, Y_train, epochs=100)  # , batch_size=100)

# evaluate the model on the test data
y_pred = model.predict(X_test)
print("y_pred = ", y_pred)
# convert softmax outputs to class labels
predictions = np.argmax(y_pred, axis=1)

# Calcualte performance metrics
f1 = sklearn.metrics.f1_score(Y_test, predictions, average='macro')
accuracy = sklearn.metrics.accuracy_score(Y_test, predictions)
print("f1-score = ", f1)
print("accuracy = ", accuracy)

# Summarize the results
print("Performance Summary:")
print(sklearn.metrics.classification_report(Y_test, predictions))
print("Confusion Matrix:")
print(sklearn.metrics.confusion_matrix(Y_test, predictions))
