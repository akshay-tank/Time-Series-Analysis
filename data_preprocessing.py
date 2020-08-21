import pandas as pd
import time

def missing_values(dataframe):
	
	print("\nTaking care of Missing Data in 'Vulnerability Type(s)' column")
	time.sleep(3)
	dataframe['Vulnerability Type(s)'].value_counts()[:1]
	dataframe["Vulnerability Type(s)"].fillna("XSS", inplace=True)
		
	return dataframe

def date_formate(dataframe):
	
	print("\nTaking care of date formate and removing rows which does not specify the date format in 'Publish Date' column")
	time.sleep(3)
	dataframe['Publish Date'] = pd.to_datetime(dataframe['Publish Date'], format="%d/%m/%Y", errors='coerce')
	dataframe = dataframe.dropna()
	
	return dataframe

