# import necessary libraries
import numpy as np
import pandas as pd
import sys
import pickle
from sklearn import preprocessing
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from math import sqrt


def load_data(database_filepath):
    """
    INPUT: 
    database_filepath (str): path to the database file
    
    OUTPUT:
    X (series, str): contains the messages
    y (dataframe): consists of all the classes
    category_names: names of the classes (target variables)
    """

    houses = pd.read_json(database_filepath, lines=True)
    le = preprocessing.LabelEncoder()
    le.fit(houses.property_type)
    houses['property_type'] = le.transform(houses['property_type'])
    
    X = houses.drop(['bedrooms'], axis=1)
    y = houses['bedrooms']
    return X, y

def build_train_and_evaluate_model(X, y):
    """
    To Create (initialize) a Model and
    To return r2 score and root mean squared error for the model
    """

    poly = PolynomialFeatures(2)
    degree2 = poly.fit_transform(X)
    X3 = sm.add_constant(degree2)
    est3 = sm.OLS(y, X3)
    est4 = est3.fit()

    r2 = est4.rsquared.round(3)
    print("R2 score for the model:", r2)
    rms = sqrt(mean_squared_error(y, np.floor(est4.predict(X3))))
    print("Root Mean Squared Error for the model:", rms)
    
    return est4

def save_model(model, model_filepath):
    """
    To save a pickle file of the trained model
    """
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y = load_data(database_filepath)
        
        print('Building model...')
        model = build_train_and_evaluate_model(X, y)
        

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')



    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'buildModel.py ../data/street_group_data_science_bedrooms_test.json regressor.pkl')


if __name__ == '__main__':
    main()