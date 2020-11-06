import pandas as pd
import numpy as np
import keras

from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.engine.saving import load_model
from keras.layers import Dense, Input, Convolution1D, MaxPool1D, Flatten
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split


def get_model(m_timepoints_per_sample):
    ecg_in = Input(shape=(m_timepoints_per_sample, 1))
    # -----------------------------------------------------------------------------------

    pooled_flow_1 = Convolution1D(8, kernel_size=3, activation="tanh")(ecg_in)
    pooled_flow_1 = MaxPool1D(pool_size=2, strides=2)(pooled_flow_1)

    pooled_flow_1 = Convolution1D(8, kernel_size=3, activation="tanh")(pooled_flow_1)
    pooled_flow_1 = MaxPool1D(pool_size=2, strides=2)(pooled_flow_1)
    # -----------------------------------------------------------------------------------

    shortcut_flow_1 = Convolution1D(8, kernel_size=4, activation="relu", strides=2)(ecg_in)
    shortcut_flow_1 = Convolution1D(8, kernel_size=4, activation="relu", strides=2)(shortcut_flow_1)

    combined_1 = keras.layers.concatenate([pooled_flow_1, shortcut_flow_1], axis=2)
    # -----------------------------------------------------------------------------------

    pooled_flow_2 = Convolution1D(24, kernel_size=3, activation="tanh")(combined_1)
    pooled_flow_2 = MaxPool1D(pool_size=2, strides=2)(pooled_flow_2)

    pooled_flow_2 = Convolution1D(32, kernel_size=3, activation="tanh")(pooled_flow_2)
    pooled_flow_2 = MaxPool1D(pool_size=2, strides=2)(pooled_flow_2)
    # -----------------------------------------------------------------------------------

    shortcut_flow_2 = Convolution1D(24, kernel_size=4, activation="relu", strides=2)(combined_1)
    shortcut_flow_2 = Convolution1D(32, kernel_size=4, activation="relu", strides=2)(shortcut_flow_2)

    combined_2 = keras.layers.concatenate([pooled_flow_2, shortcut_flow_2], axis=2)
    # -----------------------------------------------------------------------------------

    pooled_flow_3 = Convolution1D(96, kernel_size=3, activation="tanh")(combined_2)
    pooled_flow_3 = MaxPool1D(pool_size=2, strides=2)(pooled_flow_3)

    pooled_flow_3 = Convolution1D(128, kernel_size=3, activation="tanh")(pooled_flow_3)
    pooled_flow_3 = Flatten()(pooled_flow_3)

    out = Dense(1, activation="sigmoid")(pooled_flow_3)

    m_model = models.Model(inputs=ecg_in, outputs=out)
    m_model.compile(loss=losses.binary_crossentropy, optimizer=optimizers.Adam(lr=0.002), metrics=['acc'])
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
    file_path = "own_cnn_ptb.h5"

    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early = EarlyStopping(monitor="val_acc", mode="max", patience=9, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, factor=0.5, verbose=2)
    callbacks_list = [checkpoint, early, redonplat]

    model.fit(X, Y, epochs=50, batch_size=128, verbose=2, callbacks=callbacks_list, validation_split=0.1)

    model = load_model(file_path)
    pred_test = model.predict(X_test)
    pred_test = (pred_test > 0.5).astype(np.int8)

    f1 = f1_score(Y_test, pred_test)
    print("Test f1 score : ", f1)
    acc = accuracy_score(Y_test, pred_test)
    print("Test accuracy score : ", acc)

    '''
    First run: 
    Test f1 score :  0.990729736154029
    Test accuracy score :  0.9866025420817588

    Second run:
    Test f1 score :  0.9917198959072627
    Test accuracy score :  0.9879766403297836

    Third run:
    Test f1 score :  0.9919278252611586
    Test accuracy score :  0.9883201648917898

    Fourth run:
    Test f1 score :  0.9912009512485137
    Test accuracy score :  0.9872895912057712 

    Average:
    Test f1 score : 0.99139
    Test accuracy score : 0.98755
    '''
