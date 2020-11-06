
import pandas as pd
import numpy as np
import keras

from keras import optimizers, losses, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Input, Convolution1D, Flatten, Activation, MaxPooling1D
from keras.engine.saving import load_model
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split


def get_model(m_timepoints_per_sample=187):
    ecg_in = Input(shape=(m_timepoints_per_sample, 1))

    main = Convolution1D(32, kernel_size=5, activation="linear")(ecg_in)

    res = Convolution1D(32, kernel_size=5, padding="same", activation="relu")(main)
    res = Convolution1D(32, kernel_size=5, padding="same", activation="linear")(res)
    main = keras.layers.add([res, main])
    main = Activation("relu")(main)
    main = MaxPooling1D(pool_size=5, strides=2)(main)

    res = Convolution1D(32, kernel_size=5, padding="same", activation="relu")(main)
    res = Convolution1D(32, kernel_size=5, padding="same", activation="linear")(res)
    main = keras.layers.add([res, main])
    main = Activation("relu")(main)
    main = MaxPooling1D(pool_size=5, strides=2)(main)

    res = Convolution1D(32, kernel_size=5, padding="same", activation="relu")(main)
    res = Convolution1D(32, kernel_size=5, padding="same", activation="linear")(res)
    main = keras.layers.add([res, main])
    main = Activation("relu")(main)
    main = MaxPooling1D(pool_size=5, strides=2)(main)

    res = Convolution1D(32, kernel_size=5, padding="same", activation="relu")(main)
    res = Convolution1D(32, kernel_size=5, padding="same", activation="linear")(res)
    main = keras.layers.add([res, main])
    main = Activation("relu")(main)
    main = MaxPooling1D(pool_size=5, strides=2)(main)

    res = Convolution1D(32, kernel_size=5, padding="same", activation="relu")(main)
    res = Convolution1D(32, kernel_size=5, padding="same", activation="linear")(res)
    main = keras.layers.add([res, main])
    main = Activation("relu")(main)
    main = MaxPooling1D(pool_size=5, strides=2)(main)

    out = Flatten(name="last_layer")(main)
    out = Dense(32, activation="relu")(out)
    out = Dense(32, activation="linear")(out)
    out = Dense(1, activation="sigmoid")(out)

    m_model = models.Model(inputs=ecg_in, outputs=out)
    m_model.compile(loss=losses.binary_crossentropy, optimizer=optimizers.Adam(lr=0.001), metrics=['acc'])
    m_model.summary()
    return m_model


def train_model():
    # define dataset-specific constants:
    timepoints_per_sample = 187

    data_path_normal = "./input/ptbdb_normal.csv"
    data_path_abnormal = "./input/ptbdb_abnormal.csv"

    df_1 = pd.read_csv(data_path_normal, header=None)
    df_2 = pd.read_csv(data_path_abnormal, header=None)
    df = pd.concat([df_1, df_2])

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])

    Y = np.array(df_train[187].values).astype(np.int8)
    X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

    Y_test = np.array(df_test[187].values).astype(np.int8)
    X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

    # normalize without taking the zero-padding at the end into account:
    for sample_idx in range(X.shape[0]):
        first_zero_sample = timepoints_per_sample
        while X[sample_idx, first_zero_sample - 1, 0] == 0:
            first_zero_sample -= 1
        X[sample_idx, 0: first_zero_sample, 0] -= np.mean(X[sample_idx, 0: first_zero_sample, 0])
        X[sample_idx, 0: first_zero_sample, 0] /= np.std(X[sample_idx, 0: first_zero_sample, 0])

    for sample_idx in range(X_test.shape[0]):
        first_zero_sample = timepoints_per_sample
        while X_test[sample_idx, first_zero_sample - 1, 0] == 0:
            first_zero_sample -= 1
        X_test[sample_idx, 0: first_zero_sample, 0] -= np.mean(X_test[sample_idx, 0: first_zero_sample, 0])
        X_test[sample_idx, 0: first_zero_sample, 0] /= np.std(X_test[sample_idx, 0: first_zero_sample, 0])

    model = get_model(timepoints_per_sample)
    file_path = "paper_resnet_ptb.h5"

    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early = EarlyStopping(monitor="val_acc", mode="max", patience=8, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, factor=0.5, verbose=2)
    callbacks_list = [checkpoint, early, redonplat]

    model.fit(X, Y, epochs=50, batch_size=128, verbose=2, callbacks=callbacks_list, validation_split=0.2)

    model = load_model(file_path)
    pred_test = model.predict(X_test)
    pred_test[pred_test >= 0.5] = 1
    pred_test[pred_test < 0.5] = 0

    f1 = f1_score(Y_test, pred_test, average="macro")
    print("Test f1 score : ", f1)
    acc = accuracy_score(Y_test, pred_test)
    print("Test accuracy score : ", acc)

    '''
    First run:
    Test f1 score :  0.9918345081313146
    Test accuracy score :  0.9934730333218825
    
    Second run:
    Test f1 score :  0.993152674655605
    Test accuracy score :  0.9945036070079011
    
    Third run:
    Test f1 score :  0.9914146508349406
    Test accuracy score :  0.9931295087598764
    
    Fourth run:
    Test f1 score :  0.9909540368115699
    Test accuracy score :  0.9927859841978701
    
    Average:
    Test f1 score :  0.99184
    Test accuracy score :  0.99347
    '''
