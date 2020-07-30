import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
import kerastuner as kt
from kerastuner import RandomSearch, Hyperband, BayesianOptimization
from sklearn.preprocessing import MinMaxScaler


train = pd.read_csv('/train.csv')
train.set_index('sample_id', inplace=True)

train.dropna(axis=1, how='all', inplace=True)
train.replace([np.inf, -np.inf], np.nan, inplace=True)
impute = SimpleImputer(strategy='mean')
na_indexes = train.isna().sum()[train.isna().sum() > 0].sort_values(ascending=False)[train.isna().sum()[train.isna().sum() > 0].sort_values(ascending=False) < 300].index
for na_inx in tqdm(na_indexes):
    train['{}'.format(na_inx)] = impute.fit_transform(train['{}'.format(na_inx)].values.reshape(-1, 1))

train.dropna(axis=1, how='any', inplace=True)

test = pd.read_csv('/test.csv')
test.set_index('sample_id', inplace=True)

test.dropna(axis=1, how='all', inplace=True)
test.replace([np.inf, -np.inf], np.nan, inplace=True)
impute = SimpleImputer(strategy='mean')
na_indexes = test.isna().sum()[test.isna().sum() > 0].sort_values(ascending=False)[test.isna().sum()[test.isna().sum() > 0].sort_values(ascending=False) < 300].index
for na_inx in tqdm(na_indexes):
    test['{}'.format(na_inx)] = impute.fit_transform(test['{}'.format(na_inx)].values.reshape(-1, 1))
test.dropna(axis=1, how='any', inplace=True)
test = test[train.drop('y', axis=1).columns]

mx = MinMaxScaler()
for column in tqdm(train.columns[:-1]):
    mx.fit(train['{}'.format(column)].values.reshape(-1, 1))
    train['{}'.format(column)] = mx.transform(train['{}'.format(column)].values.reshape(-1, 1))
    test['{}'.format(column)] = mx.transform(test['{}'.format(column)].values.reshape(-1, 1))

def mod(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units_input', min_value=16, max_value=1024, step=32), activation='relu',
                    input_shape=(len(train.columns)-1,)
               )
         )
    
    for i in range(hp.Int('num_layers', 2, 5)):
        model.add(Dropout(0.3))
        model.add(Dense(units=hp.Int('units_' + str(i),
                                      min_value=16,
                                      max_value=1024,
                                      step=32),
                       activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate',
                      values=[1e-2, 1e-3, 1e-4])),
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model

tuner = BayesianOptimization(mod, 
                     objective='val_accuracy',
                     max_trials=10, 
                     directory='accuracy model')

tuner.search(train.drop('y',axis=1), train.y, batch_size=32, epochs=50, validation_split=0.3, verbose=1)

best_models = tuner.get_best_models(3)

submission = pd.DataFrame(best_models[0].predict(test), columns=['y'], index=test.index)
submission.to_csv('submission.csv')