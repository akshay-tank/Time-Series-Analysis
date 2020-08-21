import pandas as pd
import time
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import statsmodels.tsa.holtwinters as ets

def model_building(train_data, test_data):

	print("\nImplementing Holt Winter's Seasonal model and training the model...")
	model = ets.ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=6).fit()
	pred = model.forecast(steps=len(test_data))
	pred = pd.DataFrame(pred).set_index(test_data.index)
	time.sleep(7)

	print("\nTest results are...")
	time.sleep(3)
	print(pred)
	time.sleep(5)

	print("\nPloting results into graph and visualizing it...")
	time.sleep(4)
	figl, ax = plt.subplots(figsize=(20,9))
	#ax.plot(train_data, label="train")
	ax.plot(test_data, label="test data")
	ax.plot(pred, label="hw_test")
	plt.legend(loc="upper left")
	plt.title("Holt Winter's Seasonal Method")
	plt.ylabel("Score")
	plt.xlabel("Year")
	plt.rcParams.update({'font.size': 15})
	plt.show()

	print("\nCalculating RMSE value...")
	time.sleep(3)
	rms_hws = sqrt(mean_squared_error(test_data, pred))
	print("RMSE value for Holt Winter's Seasonal Method is : "+str(rms_hws))	

	return rms_hws
