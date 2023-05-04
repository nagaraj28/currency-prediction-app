import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import pickle

sns.set()

# Define a function to load data for a given currency
def load_data(currency):
    # Load data
    data = pd.read_csv(f"{currency}.csv", parse_dates=['Date'], dayfirst=True)
    # Convert date column to datetime data type
    data['Date'] = pd.to_datetime(data['Date'])
    # Replace missing values with NaN
    data = data.replace('null', np.nan)
    # Convert columns to float data type
    data[['Open', 'High', 'Low', 'Close']] = data[['Open', 'High', 'Low', 'Close']].astype(float)
    return data

# Define a function to train a model for a given currency
def train_model(currency):
    data = load_data(currency)
    x = data[["Open", "High", "Low"]]
    y = data["Close"]
    # Impute missing values using mean imputation
    imputer = SimpleImputer()
    x = imputer.fit_transform(x)
    y = y.to_numpy()
    y = y.reshape(-1, 1)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(xtrain, ytrain.ravel())
    ypred = model.predict(xtest)
    data = pd.DataFrame(data={"Predicted Rate": ypred.flatten()})
    pickle.dump(model, open(f'{currency}_model.pkl', 'wb'))

# Define a function to load a trained model for a given currency
def load_model(currency):
    model = pickle.load(open(f'{currency}_model.pkl', 'rb'))
    return model

# Get input from user
currency = input("Enter currency code (e.g. USD): ")

# Check if model already exists for the currency
if os.path.exists(f'{currency}_model.pkl'):
    model = load_model(currency)
else:
    train_model(currency)
    model = load_model(currency)




# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from seaborn import regression
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# import pickle
#
# sns.set()
#
# # Define a function to load data for a given currency
# def load_data(currency):
#     # Load data
#     data = pd.read_csv(f"{currency}.csv")
#     # Convert date column to datetime data type
#     data['Date'] = pd.to_datetime(data['Date'])
#     # Replace missing values with NaN
#     data = data.replace('null', np.nan)
#     # Convert columns to float data type
#     data[['Open', 'High', 'Low', 'Close']] = data[['Open', 'High', 'Low', 'Close']].astype(float)
#     return data
#
# # Define a function to train a model for a given currency
# def train_model(currency):
#     data = load_data(currency)
#     x = data[["Open", "High", "Low"]]
#     y = data["Close"]
#     x = x.to_numpy()
#     y = y.to_numpy()
#     y = y.reshape(-1, 1)
#     xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
#     model = DecisionTreeRegressor()
#     model.fit(xtrain, ytrain)
#     ypred = model.predict(xtest)
#     data = pd.DataFrame(data={"Predicted Rate": ypred.flatten()})
#     pickle.dump(model, open(f'{currency}_model.pkl', 'wb'))
#
# # Define a function to load a trained model for a given currency
# def load_model(currency):
#     model = pickle.load(open(f'{currency}_model.pkl', 'rb'))
#     return model
#
# # Get input from user
# currency = input("Enter currency code (e.g. USD): ")
#
# # Check if model already exists for the currency
# if os.path.exists(f'{currency}_model.pkl'):
#     model = load_model(currency)
# else:
#     train_model(currency)
#     model = load_model(currency)
#
#
#









# Use the model for prediction or further analysis




# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from seaborn import regression
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# import pickle
#
# sns.set()
#
# # Get input from user
# currency = input("Enter currency code (e.g. USD): ")
#
# # Load data
# data = pd.read_csv(f"{currency}.csv")
#
# # Convert date column to datetime data type
# data['Date'] = pd.to_datetime(data['Date'])
#
# # Check data types
# print(data.dtypes)
#
# # Replace missing values with NaN
# data = data.replace('null', np.nan)
#
# # Convert columns to float data type
# data[['Open', 'High', 'Low', 'Close']] = data[['Open', 'High', 'Low', 'Close']].astype(float)
#
# print(data.corr())
#
# sns.heatmap(data.corr())
# # plt.show()
#
# x = data[["Open", "High", "Low"]]
# y = data["Close"]
# x = x.to_numpy()
# y = y.to_numpy()
# y = y.reshape(-1, 1)
# xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
#
# model = DecisionTreeRegressor()
# model.fit(xtrain, ytrain)
# ypred = model.predict(xtest)
# data = pd.DataFrame(data={"Predicted Rate": ypred.flatten()})
# print(data.head())
# pickle.dump(model,open('model.pkl','wb'))
# model=pickle.load(open('model.pkl','rb'))


# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
#
# # Load the data
# data = pd.read_csv('exchange_rates.csv')
#
# # Drop any rows with missing exchange rates
# data = data.dropna(subset=['Euro/EUR', 'Japan Yen/JPY', 'Great Britain Pound/GBP', 'Australia Dollar/AUD'])
#
# # Convert date strings to datetime objects
# data['Date'] = pd.to_datetime(data['Date'])
#
# # Create a new column with the year
# data['Year'] = data['Date'].dt.year
#
# # Create separate datasets for each currency pair
# eur_usd = data[['Date', 'Euro/EUR']].copy()
# jpy_usd = data[['Date', 'Japanese Yen/USD']].copy()
# gbp_usd = data[['Date', 'Pound sterling/USD']].copy()
# aud_usd = data[['Date', 'Australian dollar/USD']].copy()
#
# # Train a linear regression model for each currency pair
# models = {}
# for currency, dataset in [('EUR/USD', eur_usd), ('JPY/USD', jpy_usd), ('GBP/USD', gbp_usd), ('AUD/USD', aud_usd)]:
#     X = dataset['Date'].astype(np.int64) // 10**9
#     y = dataset[currency]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = LinearRegression()
#     model.fit(X_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1))
#     y_pred = model.predict(X_test.values.reshape(-1, 1))
#     mse = mean_squared_error(y_test, y_pred)
#     models[currency] = {'model': model, 'mse': mse}
#
# # Make predictions for the next year
# dates = pd.date_range(start=data['Date'].max(), end=data['Date'].max() + pd.DateOffset(years=1), freq='D')
# predictions = pd.DataFrame({'Date': dates})
# for currency, model_data in models.items():
#     model = model_data['model']
#     X = dates.astype(np.int64) // 10**9
#     y_pred = model.predict(X.values.reshape(-1, 1)).flatten()
#     predictions[currency] = y_pred
#
# print(predictions.tail())
