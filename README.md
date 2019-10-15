# HSI-daily-trend-predictor

Huge credits to Analytics Vidhya and Jason for their educational resources:

	1.https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/

	2.https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

This is meant to be implemented as a practice and experiment for InvestingNote SG Active Trading Competition

This project was never intended to be applicable for current market use

# main.py

The main.py file contains a set of machine learning algorithms to predict whether daily prices of the Hang Seng Index(HSI) would rise or fall. 

Logistic Regression is the finalised model on the sole basis of cross-validation accuracy.

The yahoo.py file contains an API call to Yahoo’s API to fetch real-time data for HSI, to be used as an input for main.py

## Key Flaws:

1. Inefficient use of data (Only the ‘Open’, ‘High’, ‘Low’ data were put to use as input factors)

2. The above-mentioned factors are inherently unreliable as indicators, due to the lack of factual and predictive information it provides

3. Success in backtesting may produce tons of false positives in actual use due to overfitting

# lstm.py
The lstm.py file contains the main code used to generate a Long-Short Term Memory(LSTM) Neural Network.

To complement the short term predictions of the main file, LSTM model provided a Mid-Long Term directional estimate of HSI. Whether HSI is likely to rise or fall in the coming weeks.

I decided that a humble number of neurons and hidden layers should be used in order to refrain the model from overfitting. Precision of the model was not a significant factor to consider. However, accuracy is.

Both a picture of the performance of the model on recent data, and present data are provided within the Model Visualisations folder

The LSTM model is saved within the .h5 file

## Key Flaws:

1. Too little training samples(4000+) may limit the practical implementation of neural network

2. Much of the conceptualisation of this model was based on ‘rough-estimates’. Meaning that precision was never factored into the model generation procedure

3. Does not take into account random market elements, for example market news could randomly cause a spike or dip in HSI price

# Conclusion

Conclusively, this trading system is not fully automated and still heavily relies on human intervention and management in order to prevent large swings in trading capital. The predictions will be taken with a grain of salt, and would only partially influence my personal trading decisions SOLELY for the competition.

Ultimately, this was a good learning experience as it gave me a perspective of the sheer complexity of modelling market movements.

