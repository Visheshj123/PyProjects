from sklearn.datasets import load_boston
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


#Dictionary Dataset
boston = load_boston()

#Keys
print(boston.keys()) #data,target,feature_names, DESCR, filenmame

print(boston.data.shape) #(506,13) 506 rows by 13 columns

data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
print(data.head())

#add 'target' column to df, called price
data['PRICE'] = boston.target

print(data.describe())

#Seperate independent and dependent variable by splicing df using iloc
x,y = data.iloc[:,:-1], data.iloc[:,-1]

data_dmatrix = xgb.DMatrix(data=x,label=y)

#Split data into testing and training data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=123)

#Structure of xgboost
xg_regression = xgb.XGBRegressor(objective='reg:squarederror',colsample_bytree=0.3,learning_rate=0.1,
                                 max_depth=5, alpha=10, n_estimators=10)
#Train Model
xg_regression.fit(x_train,y_train)

#Predictions of price giving all of these parameters
preds = xg_regression.predict(x_test)


#Model Evaluation via root mean-squared error between y_test and preds
rmse = np.sqrt(mean_squared_error(y_test,preds))
print(rmse)


#Build an even better model with k-fold cross validation which allows us to use more data to build out our model

params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

cv_results = xgb.cv(params=params, dtrain=data_dmatrix,
                    nfold=3, num_boost_round=50,early_stopping_rounds=10,
                    metrics='rmse', as_pandas=True, seed=123)

print(cv_results.tail())

#Result is a rmse of $3.86 when estimating the value of a home per $1000