import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split

def main():

  file_path = 'C:/Users/spunj/Downloads/synthetic_drug_data_with_price_outliers.csv'
  print("the file: " + str(file_path))
  df = pd.read_csv(file_path)
  oc_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
  print("The dataframe: " + str(df))
  
  X = df['Price ($)'].values.reshape(-1, 1)
  #reshapes to 2d array
  print("the x: " + str(X))

  oc_svm.fit(X)
  #use correct reshaped vehicle

  #make predictions. need to reshape x_train to 2D if not already
  predictions = oc_svm.predict(X)
  print("the predictions: " + str(predictions))
  


  """#continue this later
  oc_svm.fit(X_train)
  predictions = oc_svm.predict(X_train)

  print("the predictions: " + str(predictions))"""


  #X = df.drop()

  #adjust parameters as necessary
  #ee = OneClassSVM(kernel='rbf', gamma='auto', nu= 0.1)
  
  #train model
  #ee.fit(df)

  #predictions = model.predict(df)
  #print("the predictions: " + str(predictions))

  #yhat = ee.fit_predict(df)
  #print("the yhat: " + str(yhat))

main()
