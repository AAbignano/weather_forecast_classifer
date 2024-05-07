import time

import pandas as pd
import numpy as np
import seaborn as sns

from xgboost import XGBClassifier

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import sklearn.preprocessing as preprocessing
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
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

###hyper parameter tuning here

value_counts = train_data["RainTomorrow_1"].value_counts().to_list()

param_grid = {
        'max_depth': range(5, 10),
        'n_estimators': range(80, 160, 5),
        'learning_rate': [0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 3],
        'gamma': [0, 0.01, 0.02, 0.025, 0.03, 0.04, 0.05]
}

rng = 42

start = time.time()

clf = XGBClassifier(nthread=-1,
                    scale_pos_weight=(value_counts[0]/value_counts[1]),
                    use_label_encoder=False,
                    eval_metric='mlogloss')

randomized_clf = RandomizedSearchCV(n_jobs=-1,
                                    estimator=clf,
                                    param_distributions=param_grid,
                                    scoring = 'f1',
                                    n_iter = 551, #5508
                                    cv = 3,
                                    random_state = rng)

randomized_clf.fit(x_train,y_train)

print("Best parameters: ", randomized_clf.best_params_)
print("Best score: ", randomized_clf.best_score_)
end=time.time()

print(f"{end-start} seconds elapsed")

y_pred = randomized_clf.best_estimator_.predict(x_test)

# measure accuracy
pred = randomized_clf.predict(x_test)
print('acc',metrics.accuracy_score(y_test,pred))
print('f1',metrics.f1_score(y_test,pred))
print('rec',metrics.recall_score(y_test,pred))
print('matrix\n',metrics.confusion_matrix(y_test,pred))

end=time.time()
print(f"{end-start} seconds elapsed")
