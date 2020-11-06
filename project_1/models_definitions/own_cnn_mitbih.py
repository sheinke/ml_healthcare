
import pandas as pd
import numpy as np
import keras

from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.engine.saving import load_model
from keras.layers import Dense, Input, Convolution1D, MaxPool1D,  Flatten
from sklearn.metrics import f1_score, accuracy_score


def get_model(m_timepoints_per_sample, m_num_classes):
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

    out = Dense(m_num_classes, activation=activations.softmax)(pooled_flow_3)

    m_model = models.Model(inputs=ecg_in, outputs=out)
    m_model.compile(loss=losses.sparse_categorical_crossentropy, optimizer=optimizers.Adam(lr=0.002), metrics=['acc'])
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
    file_path = "own_cnn_mitbih.h5"

    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early = EarlyStopping(monitor="val_acc", mode="max", patience=9, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, factor=0.5, verbose=2)
    callbacks_list = [checkpoint, early, redonplat]

    model.fit(X, Y, epochs=50, batch_size=128, verbose=2, callbacks=callbacks_list, validation_split=0.1)

    model = load_model(file_path)
    pred_test = model.predict(X_test)
    pred_test = np.argmax(pred_test, axis=-1)

    f1 = f1_score(Y_test, pred_test, average="macro")
    print("Test f1 score : ", f1)
    acc = accuracy_score(Y_test, pred_test)
    print("Test accuracy score : ", acc)

    '''
    First run: 
    Test f1 score :  0.9211047310741562
    Test accuracy score :  0.9857482185273159
    
    Second run:
    Test f1 score :  0.9143257550306652
    Test accuracy score :  0.9849716791522017
    
    Third run:
    Test f1 score :  0.9177145939921465
    Test accuracy score :  0.9853827882331445
    
    Fourth run:
    Test f1 score :  0.91357287708207
    Test accuracy score :  0.984469212497716
    
    Average:
    Test f1 score :  0.91668
    Test accuracy score :  0.98514
    '''
