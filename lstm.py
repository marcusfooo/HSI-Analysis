# Import Libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import sleep

# Model Parameters
days = 60
neurons = 30
epochs = 5
batch_size = 32

# Opening data file
df = pd.read_csv('HSI_171019.csv')
df.dropna(inplace=True)

# Code used to generate training and test model
def model_generation():
    # Splitting test and training data
    data_train_array = df.iloc[0:3900, :].values
    data_test_array = df.iloc[3900:4387, :].values

    # Creating test and training dataframes from np arrays
    data_train = pd.DataFrame(index=range(0,len(data_train_array)),columns=['Date', 'Close'])
    data_test = pd.DataFrame(index=range(0,len(data_test_array)),columns=['Date', 'Close'])


    for i in range(0,len(data_train)):
        data_train['Date'][i] = data_train_array[i, 0]
        data_train['Close'][i] = data_train_array[i, 5]
    data_train.index = data_train.Date
    data_train.drop('Date', axis=1, inplace=True)


    for i in range (0, len(data_test)):
        data_test['Date'][i] = data_test_array[i, 0]
        data_test['Close'][i] = data_test_array[i, 5]
    data_test.index = data_test.Date
    data_test.drop('Date', axis=1, inplace=True)


    # Scale Closing prices to between values 0,1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data_train = scaler.fit_transform(data_train)
    scaled_data_test = scaler.fit_transform(data_test)

    # Shaping training inputs and labels using past 60 day data
    x_train, y_train = [], []
    for i in range(days,len(scaled_data_train)):
        x_train.append(scaled_data_train[i-days:i,0])
        y_train.append(scaled_data_train[i,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

    # Shaping test inputs for cross-validation
    x_test = []
    for i in range(days, len(data_test_array)):
        x_test.append(scaled_data_test[i-days:i, 0])
    x_test_unscaled = np.array(data_test_array)
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


    # Main LSTM Neural Network
    model = Sequential()

    model.add(LSTM(units=neurons, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=neurons, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=neurons, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=neurons))
    model.add(Dropout(0.2))

    model.add(Dense(units = 1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    model.save('my_model.h5')


def model_testing():
    # Return rms error from cross-validation
    from keras.models import load_model
    model = load_model('my_model.h5')
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    rms=np.sqrt(np.mean(np.power((data_test_array[:, 5]-predictions),2)))
    print(rms)


    # Visual Plot of Prediction
    plt.figure(figsize=(10, 6))
    plt.plot(data_test, color='blue', label='Actual HSI Price')
    plt.plot(predictions , color='red', label='Predicted HSI Price')
    plt.title('HSI Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('HSI Price')
    plt.legend()
    plt.show()

# Actual execution of predictive model
def actual_testing():
    data_actual_array = df.iloc[0:245, :].values
    actual_plot_array = df.iloc[120:245, :].values
    data_actual = pd.DataFrame(index=range(0, len(data_actual_array)), columns=['Date', 'Close'])
    actual_plot = pd.DataFrame(index=range(0, len(actual_plot_array)), columns=['Date', 'Close'])
    for i in range(0, len(data_actual)):
        data_actual['Date'][i] = data_actual_array[i, 0]
        data_actual['Close'][i] = data_actual_array[i, 5]
    for i in range(0, len(actual_plot)):
        actual_plot['Date'][i] = actual_plot_array[i, 0]
        actual_plot['Close'][i] = actual_plot_array[i, 5]
    data_actual.index = data_actual.Date
    data_actual.drop('Date', axis=1, inplace=True)
    actual_plot.index = actual_plot.Date
    actual_plot.drop('Date', axis=1, inplace=True)
    actual_plot = np.array(actual_plot)

    # Scale Closing prices to between values 0,1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data_actual = scaler.fit_transform(data_actual)

    x_actual = []
    for i in range(days, len(data_actual_array)):
        x_actual.append(scaled_data_actual[i - days:i, 0])
    x_actual = np.array(x_actual)
    x_actual = np.reshape(x_actual, (x_actual.shape[0], x_actual.shape[1], 1))

    # Return rms error from cross-validation
    from keras.models import load_model
    model = load_model('my_model.h5')
    predictions = model.predict(x_actual)
    predictions = scaler.inverse_transform(predictions)

    # Visual Plot of Prediction
    plt.figure(figsize=(10, 6))
    plt.plot(actual_plot, color='blue', label='Actual HSI Price')
    plt.plot(predictions, color='red', label='Predicted HSI Price')
    plt.axvline(x=125, color='green', linestyle='dashed', label='Present')
    plt.title('HSI Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('HSI Price')
    plt.legend()
    plt.show()

actual_testing()
sleep(5)


