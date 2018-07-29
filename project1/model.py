import pandas as pd
from keras.layers import *
from keras.models import Sequential
from keras.callbacks import TensorBoard


def create_model():
    model = Sequential()
    model.add(Dense(50, input_dim=7, activation='relu', name='layer1'))
    model.add(Dense(100, activation='relu', name='layer2'))
    model.add(Dropout(0.3))
    model.add(Dense(100, activation='relu', name='layer3'))
    model.add(Dense(100, activation='relu', name='layer4'))
    model.add(Dense(1, activation='linear', name='output_layer'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


training_data_df = pd.read_csv("black_friday_training_scaled.csv")

X = training_data_df.drop("Purchase",axis=1).values
Y = training_data_df[["Purchase"]].values

model = create_model()
logger = TensorBoard(
    log_dir='logs',
    write_graph=True,
    histogram_freq=5
)

model.fit(
    X,
    Y,
    epochs=5,
    shuffle=True,
    verbose=2,
    callbacks=[logger]
)

testing_data_df = pd.read_csv('black_friday_testing_scaled.csv')

X_test = testing_data_df.drop('Purchase', axis=1).values
Y_test = testing_data_df[['Purchase']].values

test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
print("Mean Squared Error for test data is: {}".format(test_error_rate))

X_predict = pd.read_csv('black_friday_prediction.csv').values

predictions = model.predict(X_predict)

for prediction in predictions:
    prediction_val = prediction[0]
    prediction_val = (prediction_val + 1.5) / 0.000076968
    print("Purchase Prediction - ${}".format(prediction_val))

