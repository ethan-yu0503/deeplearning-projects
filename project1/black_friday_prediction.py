import pandas as pd
from keras.layers import *
from keras.models import Sequential
from keras.callbacks import TensorBoard


def create_model():
    model = Sequential()
    model.add(Dense(50, input_dim=7, activation='relu', name='layer1'))
    model.add(Dense(100, activation='relu', name='layer2'))
    # Adding dropout to prevent overfitting
    model.add(Dropout(0.3))
    model.add(Dense(100, activation='relu', name='layer3'))
    model.add(Dense(100, activation='relu', name='layer4'))
    # Using linear activation for a prediction model
    model.add(Dense(1, activation='linear', name='output_layer'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


training_data_df = pd.read_csv("files/black_friday_training_scaled.csv")

# The model will be able to predict purchase amounts
X = training_data_df.drop("Purchase",axis=1).values
Y = training_data_df[["Purchase"]].values

model = create_model()

# Logging onto TensorBoard in callback
logger = TensorBoard(
    log_dir='logs',
    write_graph=True,
    histogram_freq=5
)

# Training the model
model.fit(
    X,
    Y,
    epochs=30,
    shuffle=True,
    verbose=2,
    callbacks=[logger]
)

# Initializing testing
testing_data_df = pd.read_csv('files/black_friday_testing_scaled.csv')

X_test = testing_data_df.drop('Purchase', axis=1).values
Y_test = testing_data_df[['Purchase']].values

# Testing and evaluating error rate
test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
print("Mean Squared Error for test data is: {}".format(test_error_rate))


# Initializing predictions
X_predict = pd.read_csv('files/black_friday_prediction.csv').values

predictions = model.predict(X_predict)

# Printing out the purchase amount for the specified product and customer information
for prediction in predictions:
    prediction_val = prediction[0]
    prediction_val = (prediction_val + 1.5) / 0.000076968
    print("Purchase Prediction - ${}".format(prediction_val))

