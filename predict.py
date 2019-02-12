import pickle
import numpy as np
from preprocessing import Preprocessor, build_dataset

if __name__ == "__main__":

    test_data = build_dataset('test')
    preprocessor = Preprocessor()
    test_data = preprocessor.transform(test_data)

    model_file = open('model.pkl', 'rb')
    model = pickle.load(model_file)
    model_file.close()

    preds = model.predict()
    # Sales have been scaled using logarithm during preprocessing, so we need to scale them back using exponential
    preds = np.exp(preds)
