import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load training data set from CSV file
training_data_df = pd.read_csv("black_friday_training.csv")
training_data_df = training_data_df.drop(['User_ID', 'Product_ID', 'Product_Category_2', 'Product_Category_3'], axis=1)
training_data_df['Gender'] = training_data_df['Gender'].map({'F': 1, 'M':0}).astype(int)
training_data_df['City_Category'] = training_data_df['City_Category'].map({'A':1, 'B':2, 'C':3}).astype(int)
training_data_df['Stay_In_Current_City_Years'] = training_data_df['Stay_In_Current_City_Years'].map({'0': 0, '1': 1, '2': 2, '3': 3, '4+': 4}).astype(int)
training_data_df['Age'] = training_data_df['Age'].map({'0-17': 0,
                                                       '18-25': 1,
                                                       '26-35': 2,
                                                       '36-45': 3,
                                                       '46-50': 4,
                                                       '51-55': 5,
                                                       '55+': 6}
                                                      ).astype(int)

# Load testing data set from CSV file
testing_data_df = pd.read_csv("black_friday_testing.csv")
testing_data_df = testing_data_df.drop(['User_ID', 'Product_ID', 'Product_Category_2', 'Product_Category_3'], axis=1)
testing_data_df['Gender'] = testing_data_df['Gender'].map({'F': 1, 'M':0}).astype(int)
testing_data_df['City_Category'] = testing_data_df['City_Category'].map({'A':1, 'B':2, 'C':3}).astype(int)
testing_data_df['Stay_In_Current_City_Years'] = testing_data_df['Stay_In_Current_City_Years'].map({'0': 0, '1': 1, '2': 2, '3': 3, '4+': 4}).astype(int)
testing_data_df['Age'] = testing_data_df['Age'].map({'0-17': 0,
                                                       '18-25': 1,
                                                       '26-35': 2,
                                                       '36-45': 3,
                                                       '46-50': 4,
                                                       '51-55': 5,
                                                       '55+': 6}
                                                      ).astype(int)

# Data needs to be scaled to a small range like 0 to 1 for the neural
# network to work well.
scaler = MinMaxScaler(feature_range=(0,1))

# Scale both the training inputs and outputs
scaled_training = scaler.fit_transform(training_data_df)
scaled_testing = scaler.transform(testing_data_df)

# Print out the adjustment that the scaler applied to the total_earnings column of data
print("Note: Purchase values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scaler.scale_[7], scaler.min_[7]))

# Create new pandas DataFrame objects from the scaled data
scaled_training_df = pd.DataFrame(scaled_training, columns=training_data_df.columns.values)
scaled_testing_df = pd.DataFrame(scaled_testing, columns=testing_data_df.columns.values)

# Save scaled data dataframes to new CSV files
scaled_training_df.to_csv("black_friday_training_scaled.csv", index=False)
scaled_testing_df.to_csv("black_friday_testing_scaled.csv", index=False)