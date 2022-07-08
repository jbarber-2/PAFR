import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error

bob = np.load('BoB.npy')

en = np.loadtxt('energy5zkp.txt')

X = bob[0:9997] #only uses first 10000 molecules
y = en

print(X.shape)
print(y.shape)

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.9, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

def build_model(n_hidden=1, n_neurons=1000):
    model = keras.models.Sequential()
    for layer in range(n_hidden): # add in mulitple similar hidden layers with same activation function and parameters
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1)) # adds a dense layer at the end
    model.compile(loss="mean_squared_error", optimizer="Nadam") # defines loss and optimizer
    return model

model = build_model()

model.fit(X_train, y_train, epochs = 200, validation_data = (X_valid, y_valid), batch_size = 10, callbacks=[keras.callbacks.EarlyStopping(patience=10)])

y_pred = model.predict(X_test)

MAE = mean_absolute_error(y_test, y_pred)
PE = mean_absolute_percentage_error(y_test, y_pred)
RMSE =(model.evaluate(X_test, y_test))**(0.5)
R2 = r2_score(y_test, y_pred)

print('\nThe Mean Absolute error is: {:0.3f} kcal/mol'.format(MAE))
print('\nThe Mean Absolute Percent error is: {:0.3f}%'.format(PE))
print('\nThe Root Mean Squared Error is: {:0.3f}'.format(RMSE))
print('\nThe Coefficient of Determination is: {:0.3f}'.format(R2))

pred = np.concatenate(y_pred)
p = np.array(pred)
t = np.array(y_test)
error = abs(np.subtract(p, t))

with open('error_5zkq_1000.txt', 'w') as f:
    for er in error:
        f.write("%s\n" % er)
f.close()

Pred_set = np.array(bob[9997:])

pred_atoms = model.predict(Pred_set)

h = pred_atoms.tolist()

k = []
for at_en in h:
    for energy in at_en:
        k.append(energy)

with open('pred_energy_5zkp.txt', 'w') as f:
    for item in k:
        f.write("%s\n" % item)
f.close()