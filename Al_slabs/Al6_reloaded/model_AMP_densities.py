import os

import numpy as np
import matplotlib.pyplot as plt

import ase.io
from ase import Atoms, Atom, units
from ase.calculators.vasp.vasp import VaspChargeDensity
from ase.calculators.calculator import Parameters

from amp import Amp
from amp import utilities
from amp.descriptor.gaussian import Gaussian
from amp.descriptor.gaussian import FingerprintCalculator
from amp.descriptor.gaussian import NeighborlistCalculator
from amp.descriptor.cutoffs import Cosine
from amp.model.neuralnetwork import NeuralNetwork
from amp.model.neuralnetwork import NodePlot
from amp.utilities import hash_images

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from model_AMP import train_test_split

# Energies and densities are predicted using own build model

# ---------------------------------------------------------
# GLOBAL SETTINGS AND VARIABLES
# ---------------------------------------------------------

PATH_PLOT = "/home/btpl/bt702501/Dokumente/PlotsII_reloaded/"

FILENAME = 'Al6.traj'
densities = np.load('densities_{}.npy'.format(FILENAME[:-5]), allow_pickle=True)
energies = np.load('energies_{}.npy'.format(FILENAME[:-5]), allow_pickle=True)

# ---------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------

def z_score_normalization(arr_1d, mean, var):
    arr_1d = arr_1d - mean
    arr_1d = arr_1d / var
    return arr_1d 


def train(train_fingerprints, train_densities):
    network = Sequential()
    network.add(Dense(300, activation='relu', input_shape=((train_fingerprints.shape[-1],)))) 
    network.add(Dropout(0.1))
    network.add(Dense(300, activation='relu'))
    network.add(Dropout(0.1))
    network.add(Dense(300, activation='relu'))
    network.add(Dropout(0.1))
    #network.add(Dense(100, activation='relu'))
    #network.add(Dropout(0.2))
    #network.add(Dense(100, activation='relu'))
    #network.add(Dropout(0.2))
    #network.add(Dense(100, activation='relu'))
    #network.add(Dropout(0.2))
    #network.add(Dense(100, activation='relu'))
    #network.add(Dropout(0.2))
    #network.add(Dense(200, activation='relu'))
    #network.add(Dropout(0.2))
    #network.add(Dense(200, activation='tanh'))
    #network.add(Dropout(0.2))
    #network.add(Dense(200, activation='relu'))
    #network.add(Dropout(0.2))
    network.add(Dense(train_densities.shape[-1]))

    network.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    # TODO: Set separate validation dataset
    history = network.fit(train_fingerprints, train_densities, epochs=500, validation_split=0.2, batch_size=4, shuffle=True)
    
    return network, history


def predict(fp):
    return np.array(network.predict(fp))


def plot_parity(test_densities, predictions):
    # TODO: Fix to right units
    print(test_densities.shape)
    print(predictions.shape)
    for i, prediction in enumerate(predictions):
        header = 'Predictions, Actual'
        np.savetxt("{}result_{}_{}.dat".format(PATH_PLOT, FILENAME[:-5], i), 
               np.column_stack((np.array(predictions[i]), np.array(test_densities[i]))),
               header=header)
        fig = plt.figure(i, figsize=[7.04, 5.28]) 
        plt.xticks(family='cmr10')
        plt.yticks(family='cmr10')
        min_pred, max_pred = min(predictions[i]), max(predictions[i])
        min_vasp, max_vasp = min(test_densities[i]), max(test_densities[i])
        plt.plot(np.linspace(min(min_pred, min_vasp), max(max_pred, max_vasp), 1000),
                 np.linspace(min(min_pred, min_vasp), max(max_pred, max_vasp), 1000),
                 '--', alpha=1)
        plt.plot(test_densities[i], predictions[i], '.', alpha=0.4)
        plt.xlabel(r"Charge density from VASP in $\frac{e}{\AA^3}$", family='cmr10', size=14)
        plt.ylabel(r"Charge density predicted in $\frac{e}{\AA^3}$", family='cmr10', size=14)
        plt.savefig("{}Parity_{}_densities{}.png".format(PATH_PLOT, FILENAME[:-5], i), format='png', dpi=300)
        plt.close(fig)


def plot_parity_energies(test_energies, predictions):
    header = 'Predictions, Actual'
    np.savetxt("{}result_energies_{}.dat".format(PATH_PLOT,FILENAME[:-5]), 
               np.column_stack((np.array(predictions), np.array(test_energies))),
               header=header)
    fig = plt.figure(0, figsize=[7.04, 5.28])    
    plt.xticks(family='cmr10')
    plt.yticks(family='cmr10')
    min_pred, max_pred = min(predictions), max(predictions)
    min_vasp, max_vasp = min(test_energies), max(test_energies)
    plt.plot(np.linspace(min(min_pred, min_vasp), max(max_pred, max_vasp), 1000),
             np.linspace(min(min_pred, min_vasp), max(max_pred, max_vasp), 1000),
             '--', alpha=1)
    plt.plot(test_energies, predictions, '.', alpha=0.4)
    plt.xlabel(r"Energy from VASP in eV", family='cmr10', size=14)
    plt.ylabel(r"Energy predicted in eV", family='cmr10', size=14)
    plt.savefig("{}Parity_{}_energies.png".format(PATH_PLOT, FILENAME[:-5]), format='png', dpi=300)
    plt.close(fig)



def plot_loss(history):
    training_loss = np.array(history.history['loss'])
    validation_loss = np.array(history.history['val_loss']) 
    epochs = range(1, len(training_loss) + 1)   

    fig = plt.figure(-1, figsize=[7.04, 5.28])
    plt.xticks(family='cmr10')
    plt.yticks(family='cmr10')
    plt.yscale('log')
    plt.plot(epochs, training_loss, label='Training Loss')
    plt.plot(epochs, validation_loss, label='Validation Loss')
    plt.xlabel("Epoch", family='cmr10', size=14)
    plt.ylabel(r"Density in $\frac{e}{\AA^3}$", family='cmr10', size=14)
    plt.legend()
    plt.savefig("{}Losses_{}_densities.png".format(PATH_PLOT, FILENAME[:-5]), format='png', dpi=300)
    plt.close(fig)



def plot_loss_energies(history):
    training_loss = np.array(history.history['loss'])
    validation_loss = np.array(history.history['val_loss']) 
    epochs = range(1, len(training_loss) + 1)   

    fig = plt.figure(-1, figsize=[7.04, 5.28])
    plt.xticks(family='cmr10')
    plt.yticks(family='cmr10')
    plt.yscale('log')
    plt.plot(epochs, training_loss, label='Training Loss')
    plt.plot(epochs, validation_loss, label='Validation Loss')
    plt.xlabel("Epoch", family='cmr10', size=14)
    plt.ylabel(r"Energy in eV", family='cmr10', size=14)
    plt.legend()
    plt.savefig("{}Losses_{}_energies.png".format(PATH_PLOT, FILENAME[:-5]), format='png', dpi=300)
    plt.close(fig)


# Train-test split
train_images, test_images,\
train_densities, test_densities,\
train_energies, test_energies\
 = train_test_split(FILENAME, densities, energies)

train_densities = train_densities.reshape(train_densities.shape[0], -1)
test_densities = test_densities.reshape(test_densities.shape[0], -1)

train_energies = train_energies.reshape(train_energies.shape[0], -1)
test_energies = test_energies.reshape(test_energies.shape[0], -1)

# Hash images
train_images_hashed = hash_images(train_images, ordered=True)
test_images_hashed = hash_images(test_images, ordered=True)

hashed_train = []
hashed_test = []
for key, value in train_images_hashed.items():
    print("Key:",key)
    hashed_train.append(key)
for key, value in test_images_hashed.items():
    print("Key:",key)
    hashed_test.append(key)

# Calculate fingerprints
#calc = Amp.load('calc.amp')
calc = Amp(descriptor=Gaussian(),
           model=NeuralNetwork(),
           label='dens')
calc.descriptor.calculate_fingerprints(train_images_hashed)
calc.descriptor.calculate_fingerprints(test_images_hashed)

# Create dictionary with fingerprints
fingerprints_train = {}
fingerprints_test = {}
for hash_ in hashed_train:
   fingerprints_train[hash_] = calc.descriptor.fingerprints[hash_] 
for hash_ in hashed_test:
   fingerprints_test[hash_] = calc.descriptor.fingerprints[hash_] 


print("Hashed Train:", hashed_train)
print("Hashed Test:", hashed_test)

counter = 0
all_fps_train = [None] * len(hashed_train)
all_fps_test = [None] * len(hashed_test)
for i, hash_ in enumerate(hashed_train):
    for elem in fingerprints_train[hash_]:
        all_fps_train[i] = elem[1]
for i, hash_ in enumerate(hashed_test):
    for elem in fingerprints_test[hash_]:
        all_fps_test[i] = elem[1]
all_fps_train = np.array(all_fps_train)
all_fps_test = np.array(all_fps_test)


print("Train fingerprints:", all_fps_train.shape)
print("Train densities:", train_densities.shape)
print("Test fingerprints:", all_fps_test.shape)
print("Test densities:", test_densities.shape)

print("Train fingerprints:", all_fps_train.shape)
print("Train energies:", train_energies.shape)
print("Test fingerprints:", all_fps_test.shape)
print("Test energies:", test_energies.shape)

print(test_energies)


for i_col in range(all_fps_train.shape[1]):
    arr = all_fps_train[:,i_col]
    mean = np.mean(arr, axis=0)
    var = np.std(arr, axis=0)
    all_fps_train[:,i_col] = z_score_normalization(arr, mean, var)
for i_col in range(all_fps_test.shape[1]):
    arr = all_fps_test[:,i_col]
    mean = np.mean(arr, axis=0)
    var = np.std(arr, axis=0)
    all_fps_test[:,i_col] = z_score_normalization(arr, mean, var)


network_energies, history_energies = train(all_fps_train, train_energies)
predictions_energies = network_energies.predict(all_fps_test)
plot_parity_energies(test_energies, predictions_energies)
plot_loss_energies(history_energies)

network, history = train(all_fps_train, train_densities)
predictions = network.predict(all_fps_test)
plot_parity(test_densities, predictions)
plot_loss(history)
results = network.evaluate(all_fps_test, test_densities)
print(results)

