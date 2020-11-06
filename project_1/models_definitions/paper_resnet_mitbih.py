
import pandas as pd
import numpy as np
import keras

from keras import optimizers, losses, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.engine.saving import load_model
from keras.layers import Dense, Input, Convolution1D, Flatten, Activation, MaxPooling1D
from sklearn.metrics import f1_score, accuracy_score


def get_model(m_timepoints_per_sample=187, m_num_classes=5):
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
    out = Dense(m_num_classes, activation="softmax")(out)

    m_model = models.Model(inputs=ecg_in, outputs=out)
    m_model.compile(loss=losses.sparse_categorical_crossentropy, optimizer=optimizers.Adam(lr=0.001), metrics=['acc'])
    m_model.summary()
    return m_model


def train_model():
    # define dataset-specific constants:
    num_classes = 5
    timepoints_per_sample = 187

    train_path = "./input/mitbih_train.csv"
    test_path = "./input/mitbih_test.csv"

    df_train = pd.read_csv(train_path, header=None)
    df_train = df_train.sample(frac=1)
    df_test = pd.read_csv(test_path, header=None)

    Y = np.array(df_train[timepoints_per_sample].values).astype(np.int8)
    X = np.array(df_train[list(range(timepoints_per_sample))].values)[..., np.newaxis]

    Y_test = np.array(df_test[timepoints_per_sample].values).astype(np.int8)
    X_test = np.array(df_test[list(range(timepoints_per_sample))].values)[..., np.newaxis]

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

    model = get_model(timepoints_per_sample, num_classes)
    file_path = "paper_resnet_mitbih.h5"

    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early = EarlyStopping(monitor="val_acc", mode="max", patience=8, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, factor=0.5, verbose=2)
    callbacks_list = [checkpoint, early, redonplat]

    model.fit(X, Y, epochs=50, batch_size=128, verbose=2, callbacks=callbacks_list, validation_split=0.2)

    model = load_model(file_path)
    pred_test = model.predict(X_test)
    pred_test = np.argmax(pred_test, axis=-1)

    f1 = f1_score(Y_test, pred_test, average="macro")
    print("Test f1 score : ", f1)
    acc = accuracy_score(Y_test, pred_test)
    print("Test accuracy score : ", acc)

    '''
    First run:
    Test f1 score :  0.9316661944547674
    Test accuracy score :  0.9879864790791156

    Second run:
    Test f1 score :  0.9330248327810884
    Test accuracy score :  0.9877580851452585

    Third run:
    Test f1 score :  0.9272341944723722
    Test accuracy score :  0.9873469760643158

    Fourth run:
    Test f1 score :  0.9270037924287191
    Test accuracy score :  0.9878951215055728
    
    Average: 
    Test f1 score :  0.92973
    Test accuracy score :  0.98775
    '''
