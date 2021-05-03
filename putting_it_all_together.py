import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import sklearn as sklearn
import sklearn.metrics
import sklearn.preprocessing
import numpy as np
import scipy

#Divides the data into buckets for cross validation
def get_buckets(X_train,Y_train,k):
    # scikit-learn has some cross-validation functions, but we are doing
    # this to ensure you know how it is done
    num_samples, num_features = X_train.shape
    bucket_size = int(np.floor(num_samples/5))
    bucket1_size = bucket_size+int(num_samples-(bucket_size*5))
    x_buckets = []
    y_buckets = []
    bucket = X_train[:bucket1_size,:]
    x_buckets.append(bucket)
    bucket = Y_train[:bucket1_size]
    y_buckets.append(bucket)
    for i in range(1,k):
        bucket = X_train[bucket_size * i:(bucket_size * i) + bucket_size, :]
        x_buckets.append(bucket)
        bucket = Y_train[bucket_size * i:(bucket_size * i) + bucket_size]
        y_buckets.append(bucket)

    return x_buckets, y_buckets

#perform cross-validation for hyper-parameter tuning
def cross_validate(x_buckets, y_buckets, k, model):
    # scikit-learn has some cross-validation functions, but we are doing
    # this to ensure you know how it is done
    f1_scores = []
    for i in range(0, k):

        # construct train and validation for this fold
        x_val = x_buckets[i]
        y_val = y_buckets[i]
        x_train = np.zeros([0, num_features])
        y_train = np.zeros([0])
        for j in range(0, k):
            if i != j:
                x_train = np.concatenate((x_train, x_buckets[j]))
                y_train = np.concatenate((y_train, y_buckets[j]))

        # train and predict for this fold
        model.fit(x_train, y_train)
        predictions = model.predict(x_val)

        # find average metric for this fold
        f1_scores.append(sklearn.metrics.f1_score(y_val, predictions, average='macro'))

    return f1_scores

#get predictions for the test set
def get_test_set_predictions(X_train, Y_train, X_test, model):
    #train on full training set and test on test set
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    return predictions

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

from sklearn import tree
modelb = tree.DecisionTreeClassifier(criterion="gini", min_samples_split = 5)
predictions_b = get_test_set_predictions(X_train, Y_train, X_test, modelb)
f1_score_b = sklearn.metrics.f1_score(Y_test, predictions_b, average='macro')
print("test set f1 score b: ", np.mean(f1_score_b))

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

print("test set f1 score b: ", np.mean(f1_score_b))
print("f1-score = ", f1C)





print("C,DT, P-value:", get_p_value(predictionsC, predictions_b))
