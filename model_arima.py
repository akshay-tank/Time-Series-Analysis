import time
from pandas import read_csv
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
 
def model_building(train_data, test_data):

	print("\nImplementing ARIMA model and training the model...")
	history = [x for x in train_data]
	predictions = list()
	for t in range(len(test_data)):
		model = ARIMA(history, order=(4,1,0))
		model_fit = model.fit(disp=0)
		output = model_fit.forecast()
		yhat = output[0]
		predictions.append(yhat)
		obs = test_data[t]
		history.append(obs)
		#print('predicted=%f, expected=%f' % (yhat, obs))

	print("\nTest Results are...")
	time.sleep(3)
	print(predictions)
	time.sleep(5)
	
	print("\nPrinting ARIMA model's summary...")
	time.sleep(3)
	print(model_fit.summary())
	time.sleep(8)

	print("\nPloting results into graph and visualizing it...")
	time.sleep(4)
	figl, ax = plt.subplots(figsize=(20,7))
	#ax.plot(train, label="train_20")
	ax.plot(test, label="test_arima")
	ax.plot(predictions, label="pred_arima")
	plt.legend(loc="upper left")
	plt.title("Prediction through ARIMA")
	plt.ylabel("Score")
	plt.rcParams.update({'font.size': 15})
	plt.show()

	print("\nCalculating RMSE value...")
	time.sleep(3)
	rms_arima = mean_squared_error(test, predictions)
	print("RMSE value for ARIMA Method is : "+str(rms_arima))

	return rms_arima
