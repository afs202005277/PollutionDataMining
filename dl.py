import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, SimpleRNN, Dropout, Flatten, AveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from sklearn.model_selection import train_test_split


def create_model(number_of_dense, fig_name, df_train, df_test, dense=64):

    target_name = "hasHeadache"

    Y_train = df_train[target_name]
    X_train = df_train.drop([target_name], axis=1)

    Y_test = df_test[target_name]
    X_test = df_test.drop([target_name], axis=1)

    features_to_keep = X_train.columns.tolist()

    # Sort features alphabetically to guarantee consistency
    features_to_keep.sort()
    n_features = len(features_to_keep)


    # Configure tensorflow to use the GPU
    configuration = tf.compat.v1.ConfigProto()
    configuration.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=configuration)

    # Set tensor flow to run on the CPU
    tf.config.list_physical_devices('GPU')
    #tf.config.run_functions_eagerly(True)

    # introducing layers
    model = Sequential()
    model.add(Dense(dense, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(dense, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(dense // 2, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))



    # Define the optimizer
    lr_rate = 1*1e-3


    lr_scheduler_2d_order = PolynomialDecay(
        initial_learning_rate=lr_rate,
        end_learning_rate=1*1e-5,
        power=2,
        decay_steps=50
    )

    lr_schedule_const = lr_rate


    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler_2d_order)
    #optimizer = tf.keras.optimizers.SGD(learning_rate=lr_scheduler_exp)

    loss_func = tf.keras.losses.BinaryCrossentropy()


    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy'])


    model_path = '../dl/'

    n_epoch = 50
    batch_size = 32

    #early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)


    # Fit model

    history = model.fit(X_train,
                        Y_train,
                        epochs=n_epoch,
                        batch_size=batch_size,
                        validation_data=(X_test, Y_test),
                        verbose=2,
                        shuffle=False,
                        callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath=f'../dl/weights/{fig_name}/' + f'{number_of_dense}'+ '-{epoch:02d}-{val_loss:2f}.keras', monitor="val_loss", save_best_only=False, save_weights_only=False, save_freq='epoch', period=2)])



    '''
    definir save paths dos resultados
    '''
    model_name = "Vax Adverse Reactions"

    f = plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(model_name)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # Save figure
    plt.ylim(0, 1)
    loss_path = os.path.join(model_path, f"{fig_name}.jpg")
    f.savefig(loss_path)
    plt.close(f)