import pandas as pd
import numpy as np

from keras import Sequential, Model
from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout, Flatten, Lambda

from keras.layers import Conv1D, Convolution1D, Convolution2D, MaxPool1D, MaxPooling2D, GlobalMaxPool1D, AveragePooling1D, GlobalAveragePooling1D
from keras.layers import concatenate, Bidirectional, LSTM, GRU

from keras.engine.saving import load_model
from keras.utils import plot_model

from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split


def get_model(timepoints_per_sample , units_bidir=64):

    model = Sequential([
      Bidirectional(LSTM(units_bidir, return_sequences=True), input_shape=(timepoints_per_sample , 1), name='bidir1'),
      Dropout(0.5, name='dropout1'),
      Bidirectional(LSTM(units_bidir, return_sequences=False), name='bidir2'),
      Dropout(0.5, name='dropout2'),
      Dense(1, activation='sigmoid')
    ])

    opt = optimizers.Adam(0.001)  # did changing the lr help ?

    model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])
    model.summary()
    plot_model(model, show_shapes=True, show_layer_names=True, to_file="")

    return model


def train_model():

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

    # train the model

    model = get_model(timepoints_per_sample)
    file_path = "own_bidirect_bidir_lstm_ptbdb.h5"

    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early = EarlyStopping(monitor="val_acc", mode="max", patience=8, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, factor=0.3, verbose=2)
    callbacks_list = [checkpoint, early, redonplat]

    model.fit(X, Y, epochs=150, batch_size=160, verbose=2, callbacks=callbacks_list, validation_split=0.1)

    model = load_model(file_path)
    pred_test = model.predict(X_test)
    pred_test = (pred_test > 0.5).astype(np.int8)

    f1 = f1_score(Y_test, pred_test)
    print("Test f1 score : ", f1)
    acc = accuracy_score(Y_test, pred_test)
    print("Test accuracy score : ", acc)
    '''
    First run:
    Test f1 score :  0.9757834757834759
    Test accuracy score :  0.9649604946753693

    Second run:
    Test f1 score :  0.9674507008790687
    Test accuracy score :  0.9529371350051529

    Third run:
    Test f1 score :  0.9824144486692016
    Test accuracy score :  0.9745791824115424

    Fourth run:
    Test f1 score :  0.9819219790675547
    Test accuracy score :  0.9738921332875301



