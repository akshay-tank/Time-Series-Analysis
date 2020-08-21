import data_preprocessing
import model_ses
import model_hw
import model_arima
import time
import pandas as pd
from texttable import Texttable

print("\nWelcome to Forecasting of 2020's Windows Platform Vulnerability using Time Series Analysis...!!")
time.sleep(5)

# Importing the dataset
print("\n-------------------------------------Importing Dataset-------------------------------------")
time.sleep(3)
print("\nComplete Data for whole project is as under as...")

data_read = pd.read_csv("/home/akshay/Desktop/Final_Year_Project/Data/Project_data1.csv")
print(data_read)
time.sleep(7)
print('\nDescription about the data is...\n')
time.sleep(2)
print(data_read.describe())
print("\nTotal Null values in the dataset...\n")
time.sleep(2)
print(data_read.isnull().sum())
time.sleep(5)

# Data Preprocessing
print("\n-------------------------------------Data Preprocessing Started-------------------------------------")
time.sleep(2)

data_read = data_preprocessing.missing_values(data_read)
print("\n")
print(data_read)
time.sleep(2)

data_read = data_preprocessing.date_formate(data_read)
print("\n")
print(data_read)
time.sleep(2)

updated_data_read = data_read.groupby("Publish Date")['Score'].mean()


# Dividing dataset into Training Dataset and Testing Dataset
print("\nDividing dataset into Training Dataset (80% of total data) and Testing Dataset (20% of total data)...")
train_data = updated_data_read[:'2017-12-31']
test_data = updated_data_read['2018-01-01':]
time.sleep(3)


# Simple Exponential Smoothing
print("\n-------------------------------------Simple Exponential Smoothing Method-------------------------------------")
time.sleep(3)

rms_ses = model_ses.model_building(train_data, test_data)
time.sleep(3)


# Holt Winter's Seasonal Method
print("\n-------------------------------------Holt Winter's Seasonal Method-------------------------------------")
time.sleep(3)

rms_hws = model_hw.model_building(train_data, test_data)
time.sleep(3)


# ARIMA Method
print("\n-------------------------------------ARIMA Method-------------------------------------")
time.sleep(3)

rms_arima = model_arima.model_building(train_data, test_data)
time.sleep(3)


# Comparing Results
print("\nComparing results from every method by their RMSE value...")
time.sleep(3)

l = [["Method Name", "RMSE VALUE"], ["Simple Exponential Smoothing", str(rms_ses)], ["Holt Winter's Seasonal Method", str(rms_des)], ["ARIMA", str(rms_arima)]]

table = Texttable()
table.add_rows(l)
print(table.draw())

print("\nAs you can see for same data, ARIMA method gives best results...")

# Predicting Future's Vulnerability

print("\n-------------------------------------Prediction of 2020's Vulnerability through ARIMA Method-------------------------------------")

series = read_csv('/home/akshay/Desktop/Final_Year_Project/ProjectData_2020.csv')
updated_data_read_2020 = series.groupby("Publish Date")['Score'].mean()

train_2020 = updated_data_read_2020[:'2019-10-15']
test_2020 = updated_data_read_2020['2020-01-01':]

rms_arima_2020 = model_arima.model_building(train_2020, test_2020)

print("\n-------------------------------------Thank You for your precious time-------------------------------------")
