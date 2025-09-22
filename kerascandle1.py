import tensorflow as tf
import numpy as np
import pandas as pd
import config

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from keras_tuner import RandomSearch

datacsv = pd.read_csv(config.TESTAKC, sep=';', decimal=',')

x = datacsv.drop(['mh','ml'], axis=1)
y = datacsv[['mh', 'ml']]

# Разделение на train/validation
X_train, X_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Нормализация признаков
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_val = scaler.transform(X_val)

# Создание модели
input_dim = X_train.shape[1]

model = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    BatchNormalization(),
    Dropout(0.2),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.1),

    Dense(2, activation='linear')  # два выхода: maxh и minh
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',   # среднеквадратичная ошибка для обоих выходов
    metrics=['mae']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop]
)

model.summary()

# График потерь
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()

# График MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('MAE over epochs')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()

plt.tight_layout()
plt.show()

# Предсказания модели
y_pred = model.predict(X_val)

# Вывод настоящих и предсказанных значений
print("Настоящие значения (y_val):")
print(y_val)

print("Предсказанные значения (y_pred):")
print(y_pred)

print(y.describe())


def build_model(hp):
    model = Sequential()
    model.add(Dense(
        hp.Int('units_input', min_value=64, max_value=256, step=32),
        activation='relu',
        input_shape=(input_dim,)
    ))
    model.add(Dropout(hp.Float('dropout_input', 0.1, 0.5, step=0.1)))
    
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(Dense(
            hp.Int(f'units_{i}', min_value=32, max_value=128, step=32),
            activation='relu'
        ))
        model.add(Dropout(hp.Float(f'dropout_{i}', 0.1, 0.5, step=0.1)))
    
    model.add(Dense(1, activation='linear'))
    
    model.compile(
        optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='mse',
        metrics=['mae']
    )
    return model

tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='tuner_logs',
    project_name='regression_tuning'
)

tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=50, callbacks=[early_stop])
best_model = tuner.get_best_models(num_models=1)[0]