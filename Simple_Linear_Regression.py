# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:05:59 2025

@author: Mbekezeli Tshabalala
"""
# Import libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#from dmba import regressionSummary
from dmba import forward_selection, backward_elimination, stepwise_selection, AIC_score
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


# Load the data
data = pd.read_csv("C:/Users/Mbekezeli Tshabalala/OneDrive/Documents/My projects/Hex Softwares/Week 1/Project_2/housing_price_dataset/housing_price_dataset.csv")

#Inspect the dataset
data.info()
#From the results, we can see that there are no missing values
data.duplicated().sum() # no duplicates

#fixing structural error
data.columns = data.columns.str.lower().str.strip()
"""
Data preprocessing
"""
# 1. Dealing with categorical variables
data = pd.get_dummies(data, columns=['neighborhood'], drop_first=True) # Apply one-hot Encoding and avoids multicollinearity

cat_boo = data.select_dtypes('bool').columns.tolist()
data[cat_boo] = data[cat_boo].astype(int)

# 2. Standardize numerical features
num_var = ['squarefeet','yearbuilt','bedrooms','bathrooms']
scaler = MinMaxScaler()
data[num_var] = scaler.fit_transform(data[num_var])

# 3. Define features (X) and target (y)
target = data['price']
features = data.drop(columns=['price'])

# 4. Add a constant for the intercept (required for statsmodels)
features = sm.add_constant(features)

#Split the data
features_train,features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)

"""
Model fitting and Prediction
"""
# fitting the model
model = LinearRegression()
model.fit(features_train,target_train)

# Make predictions
fitted_v = model.predict(features_test)

# Evaluate the model
mae1 = mean_absolute_error(target_test, fitted_v)
mse1 = mean_squared_error(target_test, fitted_v)
rmse1 = mse1 ** 0.5
r21 = r2_score(target_test, fitted_v)

print(f"Mean Absolute Error: {mae1: .3f}")
print(f"Mean Squared Error: {mse1: .3f}")
print(f"Root Mean Squared Error: {rmse1: .3f}")
print(f"R² Score: {r21:.4f}")

"""
 Feature selection and Model fitting 2
"""
# 5. Select features using a Stepwise selection process
# Model selection: Stepwise_selection, forward_selection, and backward_elimination
def train_model(variables):
    if len(variables) == 0:
        return None
    model = LinearRegression()
    model.fit(features_train[variables], target_train)
    return model
 
def score_model(model, variables):
    if model is None:
        return AIC_score(target_train, [target_train.mean()] * len(target_train), model, df=1) 
    return AIC_score(target_train, model.predict(features_train[variables]), model)

print('Stepwise selection:')
best_model, best_variable = stepwise_selection(features_train.columns, train_model, score_model, verbose=True)
print('\nforward selection:')
best_model2, best_variable2 = forward_selection(features_train.columns, train_model, score_model, verbose=True)
print('\nBackward selection:')
best_model3, best_variable3 = backward_elimination(features_train.columns, train_model, score_model, verbose=True)

# Stepwise selection
print(f'Intercept: {best_model.intercept_: .3f}')
print('Coefficints:')
for name, coef in zip(features_train, best_model.coef_):
    print(f'{name}: {coef}')
    
data2= features_test[best_variable]

#Predicting
y_pred = best_model.predict(data2)

# Calculate errors
mae = mean_absolute_error(target_test, y_pred)
mse = mean_squared_error(target_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(target_test, y_pred)

print(f"Mean Absolute Error: {mae: .3f}")
print(f"Mean Squared Error: {mse: .3f}")
print(f"Root Mean Squared Error: {rmse: .3f}")
print(f"R² Score: {r2:.4f}\n")

    
# forward selection
print(f'Intercept: {best_model.intercept_: .3f}')
print('Coefficints:')
for name, coef in zip(features_train, best_model2.coef_):
    print(f'{name}: {coef}')
    
data3= features_test[best_variable2]

#Predicting
y_pred2 = best_model2.predict(data3)

# Calculate errors
mae1 = mean_absolute_error(target_test, y_pred2)
mse1 = mean_squared_error(target_test, y_pred2)
rmse1 = mse ** 0.5
r21 = r2_score(target_test, y_pred2)

print(f"Mean Absolute Error: {mae1: .3f}")
print(f"Mean Squared Error: {mse1: .3f}")
print(f"Root Mean Squared Error: {rmse1: .3f}")
print(f"R² Score: {r21:.4f}\n")
    
# backward selection
print(f'Intercept: {best_model.intercept_: .3f}')
print('Coefficints:')
for name, coef in zip(features_train, best_model3.coef_):
    print(f'{name}: {coef}')
    
data4= features_test[best_variable]

#Predicting
y_pred3 = best_model3.predict(data4)

# Calculate errors
mae2 = mean_absolute_error(target_test, y_pred3)
mse2 = mean_squared_error(target_test, y_pred3)
rmse2 = mse ** 0.5
r22 = r2_score(target_test, y_pred3)

print(f"Mean Absolute Error: {mae2: .3f}")
print(f"Mean Squared Error: {mse2: .3f}")
print(f"Root Mean Squared Error: {rmse2: .3f}")
print(f"R² Score: {r22:.4f}")

"""
Interpreting the regression equation
"""
# Based on these selection, the best model has squarefeet, bedrooms, bathrooms, neighborhood_Urban


