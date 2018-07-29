# Project 1: Black Friday Purchase Amount Prediction

Based on the Black Friday sales statistics (https://www.kaggle.com/mehdidag/black-friday),
this project preprocesses the dataset and predicts how much money a person with a list of
specified characteristics (gender, age range, city, years in the current city, occupation, marital
status) on a genre of products.

## Preprocessing
The elements in the dataset are converted to values between 0 to 1 for the neural network to
establish a more accurate model when training. Elements like UserID are left out of the
dataset used for training and testing for the neural network to use only the most relevant
information.

## Prediction
The prediction file creates a neural network with fully-connected layers paired with reLu,
with a linear activation function in the output layer. It then trains the model, tests it,
and uses it to make predictions on purchase amount based on user information and that genre
of products. 