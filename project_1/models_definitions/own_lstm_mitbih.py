
import pandas as pd
import numpy as np

from keras import optimizers, losses, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.engine.saving import load_model
from keras.layers import Dense, LSTM
from sklearn.metrics import f1_score, accuracy_score


def get_model(m_timepoints_per_sample, m_num_classes):
    m_model = Sequential()
    m_model.add(LSTM(40, return_sequences=True, input_shape=(m_timepoints_per_sample, 1)))
    m_model.add(LSTM(40))
    m_model.add(Dense(40, activation='tanh'))
    m_model.add(Dense(m_num_classes, activation='softmax'))

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
    file_path = "own_lstm_mitbih.h5"

    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early = EarlyStopping(monitor="val_acc", mode="max", patience=8, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, factor=0.5, verbose=2)
    callbacks_list = [checkpoint, early, redonplat]

    model.fit(X, Y, epochs=50, batch_size=400, verbose=2, callbacks=callbacks_list, validation_split=0.1)

    model = load_model(file_path)
    pred_test = model.predict(X_test)
    pred_test = np.argmax(pred_test, axis=-1)

    f1 = f1_score(Y_test, pred_test, average="macro")
    print("Test f1 score : ", f1)
    acc = accuracy_score(Y_test, pred_test)
    print("Test accuracy score : ", acc)

    '''
    First run:
    Test f1 score :  0.8830768599106025
    Test accuracy score :  0.9769322126804312
    
    Second run:
    Test f1 score :  0.8744647536245468
    Test accuracy score :  0.9767038187465741
    
    Average:
    Test f1 score :  0.87877
    Test accuracy score :  0.97682
    '''
