import time

import pandas as pd
import numpy as np
import seaborn as sns

from xgboost import XGBClassifier

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import sklearn.preprocessing as preprocessing
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Ignore the first column (row ID)
train_data = pd.read_csv('WeatherData.csv', usecols = [i for i in range(1, 23)])

# add new data that may help accuracy
train_data['Cloud_delta'] = train_data['Cloud3pm'] - train_data['Cloud9am']
train_data['Humidity_delta'] = train_data['Humidity3pm'] - train_data['Humidity9am']
train_data['Pressure_delta'] = train_data['Pressure3pm'] - train_data['Pressure9am']
train_data['Temp_delta'] = train_data['Temp3pm'] - train_data['Temp9am']
train_data['Temp_MinMax'] = train_data['MaxTemp'] - train_data['MinTemp']
train_data['WindSpeed_delta'] = train_data['WindSpeed3pm'] - train_data['WindSpeed9am']

# Show historgrams of each column
#print('Total',len(train_data.columns),'features\n',train_data.dtypes)

train_data_hist = train_data.select_dtypes(exclude = ['bool','object', 'int64'])
train_data_hist.hist(figsize = [15,15],bins = 50)
#plt.show()

# correlation map
corr = train_data.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(corr, annot=True,fmt = '.1f')
#plt.show()

# preprocess data here

train_data_hist = train_data.select_dtypes(exclude = ['bool','object', 'int64'])

for i in train_data_hist.columns:
    train_data[[i]] = preprocessing.StandardScaler().fit_transform(train_data[[i]])

features_to_transform = ['Evaporation','Humidity9am','Sunshine','Rainfall']
for i in features_to_transform:
    train_data[[i]] = preprocessing.QuantileTransformer(n_quantiles=100,output_distribution='normal',subsample=len(train_data)).fit_transform(train_data[[i]])

# one-hot encode data
onehot_categories = [ 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow', 'Location' ]
train_data = pd.get_dummies(train_data, columns=onehot_categories, drop_first=True)

# split into training and testing datasets
x_train, x_test, y_train, y_test = train_test_split(train_data.drop(columns="RainTomorrow_1"), train_data.RainTomorrow_1, test_size=0.2)

###hyper parameter tuned already

value_counts = train_data["RainTomorrow_1"].value_counts().to_list()

start = time.time()

xgb = XGBClassifier(nthread=-1,
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    eta=0.125,
                    max_depth=7,
                    n_estimators=140,
                    scale_pos_weight=(value_counts[0]/value_counts[1]))

xgb.fit(x_train, y_train)


# measure accuracy
pred = xgb.predict(x_test)
print('acc',metrics.accuracy_score(y_test,pred))
print('f1',metrics.f1_score(y_test,pred))
print('rec',metrics.recall_score(y_test,pred))
print('matrix\n',metrics.confusion_matrix(y_test,pred))

end=time.time()
print(f"{end-start} seconds elapsed")

# predict unknownData
test_data  = pd.read_csv('Assignment3-UnknownData.csv', usecols = [i for i in range(1, 22)])

test_data['Temp_MinMax'] = test_data['MaxTemp'] - test_data['MinTemp']
test_data['Temp_delta'] = test_data['Temp3pm'] - test_data['Temp9am']
test_data['Humidity_delta'] = test_data['Humidity3pm'] - test_data['Humidity9am']
test_data['WindSpeed_delta'] = test_data['WindSpeed3pm'] - test_data['WindSpeed9am']
test_data['Cloud_delta'] = test_data['Cloud3pm'] - test_data['Cloud9am']
test_data['Pressure_delta'] = test_data['Pressure3pm'] - test_data['Pressure9am']

# Show historgrams of each column

#print('Total',len(test_data.columns),'features\n',test_data.dtypes)

test_data_hist = test_data.select_dtypes(exclude = ['bool','object', 'int64'])
test_data_hist.hist(figsize = [15,15],bins = 50)
#plt.show()


# preprocess data here
for i in test_data_hist.columns:
    test_data[[i]] = preprocessing.StandardScaler().fit_transform(test_data[[i]])

features_to_transform = ['Evaporation','Humidity9am','Sunshine','Rainfall']
for i in features_to_transform:
    test_data[[i]] = preprocessing.QuantileTransformer(n_quantiles=100,output_distribution='normal',subsample=len(test_data)).fit_transform(test_data[[i]])


# one-hot encode data
onehot_categories = [ 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'Location' ]
test_data = pd.get_dummies(test_data, columns=onehot_categories, drop_first=True)

predictions = xgb.predict(test_data)

# prepare for submission
###remove index so its ready to submit without hand modification
predictions = pd.DataFrame(np.squeeze(predictions), columns=['RainTomorrow'])

predictions.rename(columns = {'RainTomorrow':'predict-RainTomorrow'}, inplace = True)

submission = pd.read_csv('UnknownData.csv', usecols = ["row ID"])

submission = submission.join(predictions);

submission.to_csv("weather_predictions.csv", index=False)

