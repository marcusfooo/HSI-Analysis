# Import Libraries
from sklearn.preprocessing import MinMaxScaler
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
df = pd.read_csv('SNP.csv')
df.dropna(inplace=True)

def actual_testing():
    data_actual_array = df.iloc[4280:4731, :].values
    actual_plot_array = df.iloc[4400:4731, :].values
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
    model = load_model('snp_model.h5')
    predictions = model.predict(x_actual)
    predictions = scaler.inverse_transform(predictions)

    # Visual Plot of Prediction
    plt.figure(figsize=(10, 6))
    plt.plot(actual_plot, color='blue', label='Actual S&P 500 Price')
    plt.plot(predictions, color='red', label='Predicted S&P 500  Price')
    plt.axvline(x=330, color='green', linestyle='dashed', label='Present')
    plt.title('S&P 500  Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('S&P 500  Price')
    plt.legend()
    plt.show()

actual_testing()
sleep(5)