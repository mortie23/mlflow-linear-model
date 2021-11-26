# Databricks notebook source
# MAGIC %md
# MAGIC # Simple linear model

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

# %%
'''
Demo of basic linear regression model
Saving the output as a pickle file
Predicting a single point
'''

import pandas
from sklearn import model_selection
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import mlflow
from mlflow.models.signature import infer_signature

# COMMAND ----------

# %%
# Read in some dummy data from a file
path = "dummydata.csv"
df = pandas.read_csv(path)
array = df.values
X = array[:, 0:1]
Y = array[:, 1]
# Plot the dummy data
plt.scatter(X, Y, s=50, alpha=0.5)

# COMMAND ----------

# %%
# split data into training and test sets
test_size = 0.33
seed = 7
data_X_train, data_X_test, data_Y_train, data_Y_test = model_selection.train_test_split(
    X, Y, test_size=test_size, random_state=seed)

# Plot the training and test data
plt.scatter(data_X_train, data_Y_train, s=50, alpha=0.5)
plt.scatter(data_X_test, data_Y_test, s=150, alpha=0.5)

# COMMAND ----------

# %%
with mlflow.start_run() as run:
  run_id = run.info.run_id
  # Automatically capture the model's parameters, metrics, artifacts,
  # and source code with the `autolog()` function
  mlflow.sklearn.autolog()
  
  # Create linear regression object
  regr = linear_model.LinearRegression()
  # Train the model using the training sets
  regr.fit(data_X_train, data_Y_train)
  # Make predictions using the testing set
  data_Y_pred = regr.predict(data_X_test)

  r2score = r2_score(data_Y_test, data_Y_pred)
  mse = mean_squared_error(data_Y_test, data_Y_pred)
  # y = 17.739x - 2.1007
  # The coefficient and intercept of the linear
  print("Coefficients: ", regr.coef_[0])
  print("Intercept: ", regr.intercept_)
  print("Mean squared error: %.2f" % mse)
  print("Coefficient of determination: %.2f" % r2score)
  mlflow.log_metric("coef_", regr.coef_[0])
  mlflow.log_metric("intercept_", regr.intercept_)
  
  input_example = np.array([[0.8]])
  signature = infer_signature(data_X_train, regr.predict(data_X_train))
  mlflow.sklearn.log_model(regr, "linearmodel", signature=signature, input_example=input_example)

  # save the model to disk
  modelpath = "/dbfs/mlflow/lineartest/linearmodel"+run_id
  mlflow.sklearn.save_model(regr, modelpath)

  # Plot outputs
  plt.scatter(data_X_test, data_Y_test, alpha=0.5, s=150)
  plt.plot(data_X_test, data_Y_pred, color='red', linewidth=3)
  image = plt.show()

# COMMAND ----------

# %%
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(data_X_test, data_Y_test)
print(result)

# COMMAND ----------

# %%
# Predict single value
loaded_model.predict(np.array([[0.8]]))
