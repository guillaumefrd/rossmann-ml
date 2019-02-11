import pickle

def predict(X, classifier_filename):
        """Predict method
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The features.
            
        classifier_filename : string
                              Name of the pickle file containing 
                              the trained classifier

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            The predicted target.
        """
    classifier_pkl = open(classifier_filename, 'rb')
    clf = pickle.load(classifier_pkl)
    print("Loaded classifier : ", clf)
    return clf.predict(X)