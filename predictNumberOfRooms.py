# import necessary libraries
from sklearn.externals import joblib
import sys
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import numpy as np
import pandas as pd


def transform_test_data(X):
    """
    INPUT: 
    X (dataframe): Test Data
    
    OUTPUT:
    X3 (dataframe): Polynomially Transformed data 
    """
    poly = PolynomialFeatures(2)
    degree2 = poly.fit_transform(X)
    X3 = sm.add_constant(degree2)
    return X3

def load_model(model_filepath):
    """
    To load the saved model
    """
    saved_model = joblib.load(model_filepath)
    return saved_model

def main():
    if len(sys.argv) == 3:
        test_data_filepath, model_filepath = sys.argv[1:]
        print('Loading test data...\n    DATABASE: {}'.format(test_data_filepath))
        X = pd.read_json(test_data_filepath)
        le = preprocessing.LabelEncoder()
        le.fit(X.property_type)
        X['property_type'] = le.transform(X['property_type'])
        test_data = transform_test_data(X)
        saved_model = load_model(model_filepath)
        r2 = saved_model.rsquared.round(3)
        print("R2 score for the model:", r2)
        predictions = np.floor(saved_model.predict(test_data))
        print("Predictions of the model are:", predictions)


        

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'predictNumberOfRooms.py ../data/testData.json classifier.pkl')


if __name__ == '__main__':
    main()