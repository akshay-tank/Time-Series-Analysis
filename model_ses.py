import pandas as pd
import time
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import statsmodels.tsa.holtwinters as ets

def model_building(train_data, test_data):

	print("\nImplementing Simple Exponential Smoothing model and training the model...")
	ses_train = ets.ExponentialSmoothing(train_data, trend=None, damped=False, seasonal=None).fit()
	ses_test = ses_train.forecast(steps=len(test_data))
	ses_test = pd.DataFrame(ses_test).set_index(test_data.index)
	time.sleep(7)

	print("\nTest results are...")
	time.sleep(3)	
	print(ses_test)
	time.sleep(5)

	print("\nPloting results into graph and visualizing it...")
	time.sleep(4)
	figl, ax = plt.subplots(figsize=(20,7))
	#ax.plot(train_data, label="train")
	ax.plot(test_data, label="test data")
	ax.plot(ses_test, label="ses-test prediction")
	plt.legend(loc="upper left")
	plt.title("Exponential Smoothing Method")
	plt.ylabel("Score")
	plt.xlabel("Year")
	plt.rcParams.update({'font.size': 15})
	plt.show()
	
	print("\nCalculating RMSE value...")
	time.sleep(3)
	rms_ses = sqrt(mean_squared_error(test_data, ses_test))
	print("RMSE value for Simple Exponential Smoothing is : "+str(rms_ses))
	
	return rms_ses
