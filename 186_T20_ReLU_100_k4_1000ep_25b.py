#186 machine T20 all ReLU ran 100 times for k fold = 4 (bagging code) with 1000 epochs and batchsize 25


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

pd.set_option('max_columns', 100)
###################################
def norm(x, train_dataset):
    train_stats = train_dataset.describe().transpose()
    return (x - train_stats['mean']) / train_stats['std'].replace(to_replace=0, value=1)


# this is the model used in Clyde's bagging code

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


epochs = 1000
batch_size = 25


###################################

CSN_path = './'
#CSN_path = './Data/'

def load_CSN_data():
    csv_path = CSN_path + "Master_List_LCPLCP.csv"
    return pd.read_csv(csv_path)

CSN = load_CSN_data()

CSN_new_err = CSN['Error'][-18:]

CSN = CSN.drop(['Example ID', 'Source', 'Figure ID', 'Data Provider', 'PI',
       'Date Received', 'Data Measurment Published', 'Prior Exposure', 'Comments', 'Error'], axis=1)

tsize = CSN.shape[0]//10

CSN_new = pd.get_dummies(CSN)

CSN_new['Surface Area per Liter'] = CSN_new['Surface Area (NMC) (m2/g)'] * CSN_new['Concentration (mg/L)']
CSN_new = CSN_new.drop(['Surface Area (NMC) (m2/g)'], axis=1)

CSN_new['log Concentration'] = np.log10(CSN_new['Concentration (mg/L)'] + 1e-9)
CSN_new = CSN_new.drop(['Concentration (mg/L)'], axis=1)

CSN_new_A = CSN_new[-18:] #assign the last 18 examples for 18 test cases for a different array

CSN_prepared_B = CSN_new[:-18] # Removing the last 18
CSN_hold_1 = sklearn.utils.shuffle(CSN_prepared_B, random_state=5946) #shuffling the 206 data examples as Clyde did
tsize1 = CSN_hold_1.shape[0]//10 # decide the size of the test examples which is 20 - clyde did it

CSN_prepared_B = CSN_hold_1[:-tsize1] #Remove that 20 data examples from the bottom of the list
CSN_test = CSN_hold_1[-tsize1:] # assign that 20 data examples whcih we will use later
###################################
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
out = np.zeros((100, 1, 2, 20))

for i in range(100):

    np.random.seed(i)
    CSN_hold  = sklearn.utils.shuffle(CSN_prepared_B, random_state=np.random.randint(0, 100000))
    out[i] = np.array([ikfold(i, CSN_hold, CSN_test, np.random.randint(0, 100000, i))
                 for i in [4]])

    print((i+1), '% complete')
    out.dump('86_T20_1000epochs_25batchsize.pkl')
###################################

out.dump('186_T20_1000epochs_25batchsize.pkl')


###################################

