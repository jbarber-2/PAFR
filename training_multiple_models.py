import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

bob = np.load('BoB.npy')

en = np.loadtxt('true_energy_5zkp.txt')

X = bob
y = en

all_MAE = []

for i in range(10):
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.9, random_state=i)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1000, activation="relu"))
    model.add(keras.layers.Dense(1))
    model.compile(loss="mean_squared_error", optimizer="Nadam")

    model.fit(X_train, y_train, epochs=200, validation_data=(X_valid, y_valid), batch_size=10,
              callbacks=[keras.callbacks.EarlyStopping(patience=10)])

    y_pred = model.predict(X_test)

    MAE = mean_absolute_error(y_test, y_pred)

    pred = np.concatenate(y_pred)
    p = np.array(pred)
    t = np.array(y_test)
    error = abs(np.subtract(p, t))

    with open('error_5zkq_5000_' + str(i) + '.txt', 'w') as f:
        for er in error:
            f.write("%s\n" % er)
        f.close()

    all_MAE.append(MAE)

print(all_MAE)
print(np.mean(all_MAE))
print(np.std(all_MAE))
