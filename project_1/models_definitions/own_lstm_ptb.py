
import pandas as pd
import numpy as np

from keras import optimizers, losses, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.engine.saving import load_model
from keras.layers import Dense, LSTM
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split


def get_model(m_timepoints_per_sample):
    m_model = Sequential()
    m_model.add(LSTM(40, return_sequences=True, input_shape=(m_timepoints_per_sample, 1)))
    m_model.add(LSTM(40))
    m_model.add(Dense(50, activation='relu'))
    m_model.add(Dense(50, activation='relu'))
    m_model.add(Dense(1, activation='sigmoid'))

    m_model.compile(loss=losses.binary_crossentropy, optimizer=optimizers.Adam(lr=0.0008), metrics=['acc'])
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
    file_path = "own_lstm_ptb.h5"

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
    Test f1 score :  0.9127516778523491
    Test accuracy score :  0.8704912401236689

    Second run:
    Test f1 score :  0.9123770304278197
    Test accuracy score :  0.8684300927516317
    
    Average:
    Test f1 score : 0.91256
    Test accuracy score : 0.86946
    '''
