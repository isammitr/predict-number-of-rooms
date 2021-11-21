# Predict the Number of Bedrooms

## Project Overview
Take home assignment for the [Street Group](https://www.streetgroup.co.uk/) to create a model to predict the number of bedrooms from the given dataset. The features include the type of property, floor area of the house, number of habitable rooms, estimated market prices, etc. This is clearly not a classification problem, but a regression problem since the number of bedrooms is a continuous variable. If I had more time I would explore the dataset more and check the statistically significant features and re-train the model by removing the statistically insignificant features.

## Project Components
1. **Data and Model Exploration.ipynb**: This jupyter notebook includes the exploration of the dataset, pre-processing as required and exploration of multiple regression models along with the comparision of their metrics.


2. **buildModel.py**: This script contains the data loading pipeline which
- Loads the data
- Builds, trains, and Evaluate the polynomial regression model
- Saves the model into a pickle file which can be found [here](https://drive.google.com/drive/folders/1LFIpgMpwTx8NRHirO9_Qombjhxwuv4Lr?usp=sharing)

3. **predictNumberOfRooms.py**: This script contains the testing pipeline which
- Pre-processes the test data 
- Transforms the test data into polynomial form
- Loads the saved model
- Predicts the number of rooms on the test data

## Instructions:

Run the following commands in the project's root directory to set up your database and model.

1. To build and train the polynomial regression model and save it: `python buildModel.py street_group_data_science_bedrooms_test.json model1.pkl`
2. To test the saved model with a sample test dataset in `json` format with columns `property_type`,	`total_floor_area`,	`number_habitable_rooms`,	`number_heated_rooms`, 	`estimated_min_price`,	`estimated_max_price`,	`latitude`,	`longitude`: `python predictNumberOfRooms.py test_data.json model1.pkl`
