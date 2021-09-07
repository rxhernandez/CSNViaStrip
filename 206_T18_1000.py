#206 machine T18 all ReLU ran 1000 times for all k folds (bagging code)


import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
from keras import optimizers
from tensorflow.keras import layers

import numpy as np
np.set_printoptions(precision=5)
import matplotlib.pyplot as plt
import pandas as pd
pd.options.display.float_format = '{:,.5f}'.format
import scipy as sp
from scipy import stats
import sklearn

import time

###################################

def norm(x, train_dataset):
    train_stats = train_dataset.describe().transpose()
    return (x - train_stats['mean']) / train_stats['std'].replace(to_replace=0, value=1)

def build_model(data):
    model = keras.Sequential()
    model.add(keras.layers.Dense(5, activation=tf.nn.relu, input_shape=(data.shape[1],)))
    for k in range(3-1):
        model.add(keras.layers.Dense(5, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.relu))

    model.compile(loss='mse',
                optimizer=keras.optimizers.Adam(),
                metrics=['mae'])

    return model

epochs = 100
batch_size = 1

###################################

CSN_path = './'

def load_CSN_data():
    csv_path = CSN_path + "Master_List_LCPLCP.csv"
    return pd.read_csv(csv_path)

CSN = load_CSN_data()

CSN = CSN.drop(['Example ID', 'Source', 'Figure ID', 'Data Provider', 'PI',
       'Date Received', 'Data Measurment Published', 'Prior Exposure', 'Comments', 'Error'], axis=1)

tsize = CSN.shape[0]//10

CSN_prepared = pd.get_dummies(CSN)

CSN_prepared['Surface Area per Liter'] = CSN_prepared['Surface Area (NMC) (m2/g)'] * CSN_prepared['Concentration (mg/L)']
CSN_prepared = CSN_prepared.drop(['Surface Area (NMC) (m2/g)'], axis=1)

CSN_prepared['log Concentration'] = np.log10(CSN_prepared['Concentration (mg/L)'] + 1e-9)
CSN_prepared = CSN_prepared.drop(['Concentration (mg/L)'], axis=1)

CSN_prepared = CSN_prepared[:-18]

###################################

CSN_path = './'

def load_CSN_data():
    csv_path = CSN_path + "Master_List_LCPLCP.csv"
    return pd.read_csv(csv_path)

CSN = load_CSN_data()

CSN_new_err = CSN['Error'][-18:]

CSN = CSN.drop(['Example ID', 'Source', 'Figure ID', 'Data Provider', 'PI',
       'Date Received', 'Data Measurment Published', 'Prior Exposure', 'Comments', 'Error'], axis=1)

#tsize = CSN.shape[0]//10

CSN_new = pd.get_dummies(CSN)

CSN_new['Surface Area per Liter'] = CSN_new['Surface Area (NMC) (m2/g)'] * CSN_new['Concentration (mg/L)']
CSN_new = CSN_new.drop(['Surface Area (NMC) (m2/g)'], axis=1)

CSN_new['log Concentration'] = np.log10(CSN_new['Concentration (mg/L)'] + 1e-9)
CSN_new = CSN_new.drop(['Concentration (mg/L)'], axis=1)

CSN_new = CSN_new[-18:]
#################################

def ikfold(k, data, test, s):

    CSN_shuf = data#sklearn.utils.shuffle(data, random_state=25)
    valn = data.shape[0]//k
    vscores = []
    tscores = []
    vloss = []
    tloss = []
    val_pred = np.zeros((k, test.shape[0]))
    weights = np.zeros(k)

    norm_train = data.drop(['Viability Fraction '], axis=1)

    nCSN_test = norm(test.drop(['Viability Fraction '], axis=1),
    norm_train)

    start_time = time.time()

    for i in range(k):

        if k == 1:
            val = CSN_shuf
            train = CSN_shuf
        else:
            val = CSN_shuf[valn*i:valn*(i+1)]
            train = CSN_shuf[valn*(i+1):].append(CSN_shuf[:valn*i])

        train_f = train.drop(['Viability Fraction '], axis=1)
        val_f = val.drop(['Viability Fraction '], axis=1)

        ntrain_f = norm(train_f, norm_train)
        ntrain_l = train['Viability Fraction ']
        nval_f = norm(val_f, norm_train)
        nval_l = val['Viability Fraction ']

        tf.keras.backend.clear_session()
        unique_seed = s[i]
        np.random.seed(unique_seed)
        tf.random.set_seed(unique_seed)

        model = build_model(ntrain_f)

        history = model.fit(ntrain_f,
        ntrain_l,
        validation_data=(nval_f, nval_l),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0)

        #plot_mae(history)

        val_pred[i] = model.predict(nCSN_test).flatten()
        weights[i] = 1/history.history['val_mae'][-1]

    WMean = np.average(val_pred, axis=0, weights=weights)
    Werr = np.sqrt(np.average((WMean-val_pred)**2, weights=weights, axis=0))/np.sqrt(k)
    print(k, time.time() - start_time)

    return WMean, Werr

###################################

out = np.zeros((1000, 4, 2, 18))

for i in range(1000):

    np.random.seed(i)
    CSN_hold  = sklearn.utils.shuffle(CSN_prepared, random_state=np.random.randint(0, 100000))
    out[i] = np.array([ikfold(i, CSN_hold, CSN_new, np.random.randint(0, 100000, i))
                 for i in [1, 4, 7, 10]])

    print((i+1), '% complete')
    out.dump('206_T18_1000.pkl')

###################################

out.dump('206_T18_1000.pkl')

##################################


