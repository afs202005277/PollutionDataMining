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
from tensorflow.keras.layers import BatchNormalization
import keras_tuner as kt

def build_model(hp):
    model = Sequential()

    # Input layer
    model.add(Dense(units=hp.Int('units_input', min_value=32, max_value=512, step=32),
                    input_dim=145, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_input', min_value=0.1, max_value=0.5, step=0.1)))

    for i in range(hp.Int('num_layers', 1, 5)):  # Number of hidden layers to try
        model.add(Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Learning rate tuning
    learning_rate = hp.Float('lr', min_value=1e-5, max_value=1e-2, sampling='LOG')

    # Define optimizer with learning rate tuning
    optimizer = Adam(learning_rate=learning_rate)

    # Compile the model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def create_model(fig_name, df_train, df_test, dense=64):

    target_name = "hasHeadache"

    Y_train = df_train[target_name]
    X_train = df_train.drop([target_name], axis=1).astype(np.float32)

    Y_test = df_test[target_name]
    X_test = df_test.drop([target_name], axis=1).astype(np.float32)

    features_to_keep = X_train.columns.tolist()

    # Sort features alphabetically to guarantee consistency
    features_to_keep.sort()
    n_features = len(features_to_keep)

    # Configure tensorflow to use the GPU
    configuration = tf.compat.v1.ConfigProto()
    configuration.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=configuration)

    # introducing layers
    model = Sequential()
    
    model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    
    model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(dense, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    
    model.add(Dense(dense * 2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    
    model.add(Dense(dense // 2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    
    model.add(Dense(dense // 4, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    
    model.add(Dense(1, activation='sigmoid'))

    # Learning rate scheduler using Polynomial Decay
    lr_rate = 1e-2
    lr_scheduler_2d_order = PolynomialDecay(
        initial_learning_rate=lr_rate,
        end_learning_rate=5e-5,
        power=2,
        decay_steps=100
    )

    # Define the optimizer with the learning rate scheduler
    optimizer = Adam(learning_rate=lr_scheduler_2d_order)

    # Loss function
    loss_func = tf.keras.losses.BinaryCrossentropy()

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy'])

    # Model callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Fit model
    history = model.fit(X_train,
                        Y_train,
                        epochs=50,
                        batch_size=32,
                        validation_data=(X_test, Y_test),
                        verbose=2,
                        shuffle=False,
                        callbacks=[
                            early_stopping,
                            tf.keras.callbacks.ModelCheckpoint(
                                filepath=f'../dl/weights/{fig_name}/' + '{epoch:02d}-{val_loss:.2f}.keras',
                                monitor="val_loss",
                                save_best_only=False,
                                save_weights_only=False,
                                save_freq='epoch'
                            )
                        ])

    # Plotting the loss and saving the figure
    model_name = "Vax Adverse Reactions"
    f = plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(model_name)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Save figure
    loss_path = os.path.join('dl/', f"{fig_name}.jpg")
    f.savefig(loss_path)
    plt.close(f)