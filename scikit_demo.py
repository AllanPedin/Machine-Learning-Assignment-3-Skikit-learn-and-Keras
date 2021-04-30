import sklearn
import scipy
import numpy as np

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


# Main Method to compare performance of different classifiers and hyper-parameters
if __name__ == '__main__':
    #load the iris data
    file_path = 'iris.data'
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:,:-1]
    Y = data[:,-1]

    #get a test set
    num_samples, num_features = X.shape
    train_cutoff = int(num_samples * 0.8)
    X,Y = sklearn.utils.shuffle(X,Y)
    X_train = X[:train_cutoff,:]
    Y_train = Y[:train_cutoff]
    X_test = X[train_cutoff:,:]
    Y_test = Y[train_cutoff:]

    #divide the data into buckets
    k = 5
    x_buckets, y_buckets = get_buckets(X_train, Y_train, k)

    #create a decision tree model
    from sklearn import tree
    decision_tree_md5 = tree.DecisionTreeClassifier(max_depth=1)
    decision_tree_md100 = tree.DecisionTreeClassifier(max_depth=100)

    #get cv results for a model
    print ("Cross-validation Performance")
    f1_scores_a = cross_validate(x_buckets, y_buckets, k, decision_tree_md5)
    f1_scores_b = cross_validate(x_buckets, y_buckets, k, decision_tree_md100)
    print("mean f1 a: ", np.mean(f1_scores_a))
    print("mean f1 b: ", np.mean(f1_scores_b))
    p = get_p_value(f1_scores_a, f1_scores_b)
    print("p-value comparing cross-validation performance of a vs. b = ",p)
    #compare p to alpha and make a conlcusion


    #get test set predictions to compare two models on the test set
    print("\nTest Set Performance")
    predictions_a = get_test_set_predictions(X_train, Y_train, X_test, decision_tree_md5)
    predictions_b = get_test_set_predictions(X_train, Y_train, X_test, decision_tree_md100)
    f1_score_a = sklearn.metrics.f1_score(Y_test, predictions_a, average='macro')
    f1_score_b = sklearn.metrics.f1_score(Y_test, predictions_a, average='macro')
    print("test set f1 score a: ", np.mean(f1_score_a))
    print("test set f1 score b: ", np.mean(f1_score_b))
    p = get_p_value(predictions_a, predictions_b)
    print ("p-value comparing test set performance of a vs. b = ", p)
    #compare your p-value to alpha to make a conclusion


    #######
    # Scikit-learn has implementations of the algorithms we've talked about in class (and more)
    # With the knowledge you've learned in this class, you should be able to intelligently decide
    # on what classifier to use, and to understand the different parameter options for these
    # algorithms
    ######
    #K nearest neighbors
    from sklearn import neighbors
    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    #Linear classifier (ridge regression)
    from sklearn import linear_model
    ridge_regression = linear_model.Ridge(alpha=0.5)
    #neural network
    from sklearn import neural_network
    ann = neural_network.MLPClassifier()
    #kernel SVM with an rbf kernel
    from sklearn import svm
    kernel_svm = svm.SVC(kernel='rbf')
    #decision tree
    from sklearn import tree
    decision_tree = tree.DecisionTreeClassifier()
    #adaboost
    from sklearn import ensemble
    adaboost = ensemble.AdaBoostClassifier()


