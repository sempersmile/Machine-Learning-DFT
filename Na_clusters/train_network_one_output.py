# Fingerprinting input:
# For each grid point the distance of the grid point to each atom is calculated using the euclidean metric
# These distances are then fed into the NN, i.e. the input vector will be of dimension n_atoms
# The output is a density value at the specific grid-point
# There will be n_grid_points neuronal networks in total

import os
import logging
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

logging.basicConfig(level=logging.INFO)

RECALCULATE = False # Choose whether to recalculate the distances between grid-points and atoms

#######################################
# Convert Cube-File
#######################################

# N_GRID_POINTS = 9261

parameter_set='Parameter_Set_06/'
path_data = "/tp_leppert/bt702501/Calculated_Data/"+parameter_set

def convert_cube_file(path_data):#,n_grid_points=N_GRID_POINTS):
    # Read number of atoms per structure from parameter-file
    f = open(path_data+"program_parameters.dat", "r")
    params_string = f.readlines()
    n_atoms = int(params_string[2][-2])# WARNING: This line would not be valid for more than 9 Atoms per structure!
    f.close()

    ## Read number of grid points from Grid_Index0000-file
    #n_grid_points = 0
    #with open(path_data+'train0/'+'Grid_Index0000.dat') as f:
    #    n_grid_points = sum(1 for _ in f) - 2

    # Create array for grid with positions of grid_points
    f_cube = open(path_data+"/train0/dens_orbU00.cube")
    cube_all_lines = f_cube.readlines()
    n_x = int(float(cube_all_lines[3][2:4]))
    n_y = int(float(cube_all_lines[4][2:4]))
    n_z = int(float(cube_all_lines[5][2:4]))
    f_cube.close()
    n_grid_points = n_x * n_y * n_z
    
    f_conf = open(path_data+"/train0/BTDFT_guess.conf", "r")
    conf_all_lines = f_conf.readlines()
    start_index = conf_all_lines[9].find("=") + 2
    end_index = conf_all_lines[9].find("b") - 1
    grid_spacing = float(conf_all_lines[9][start_index:end_index])
    f_conf.close()
    
    grid_coords = np.zeros((n_grid_points, 3))
    start_loop = -(n_x-1)/2 * grid_spacing # Assumption that all 3 dimensions in space are equal
    end_loop = (n_x-1)/2 * grid_spacing + grid_spacing
    point_index = 0
    for x in np.arange(start_loop, end_loop, grid_spacing):
        for y in np.arange(start_loop, end_loop, grid_spacing):
            for z in np.arange(start_loop, end_loop, grid_spacing):
                grid_coords[point_index][0] = x
                grid_coords[point_index][1] = y
                grid_coords[point_index][2] = z
                point_index += 1
    
    # Create list with all folders 
    subfolders_data = []
    n_structures = 0
    for obj in os.listdir(path_data):
        if os.path.isdir(path_data+obj):
            subfolders_data.append(obj) 
            n_structures += 1           
    subfolders_data.sort()

    structure_coords = np.zeros((n_structures, n_atoms, 3))
    data_x = np.zeros((n_structures, n_grid_points, n_atoms))
    data_y = np.zeros((n_structures, n_grid_points, 1))

    for i, directory in enumerate(subfolders_data):
        # Read densities at grid-points from .cube-file and coordinates from .npy-file
        f_read = open(path_data+directory+"/dens_orbU00.cube", "r")
        all_lines = f_read.readlines()
        f_read.close()
        all_lines_split = [line.split('    ') for line in all_lines[6+n_atoms:]]
        all_lines_flattened = [y.replace('\n','') for x in all_lines_split for y in x]
        data_single_structure = np.array((all_lines_flattened,)) 
        data_single_structure = data_single_structure[data_single_structure!='']
        data_single_structure = data_single_structure.astype(np.float64)
        # WARNING: Note that the order of the points might be odd 
        #data_single_structure = data_single_structure[data_single_structure!=0.0] 
        data_y[i,:,:] = data_single_structure.reshape(-1,1)

        structure_coords[i,:,:] = np.load(path_data+directory+'/coordinates.npy')
        
        # Calculate distances
        for i_grid, grid_point in enumerate(grid_coords):
            for i_atom, atom_coords in enumerate(structure_coords[i,:,:]):
                data_x[i,i_grid,i_atom] = np.linalg.norm(atom_coords-grid_coords)
        
        if int((i/n_structures)*100) - (i/n_structures)*100 < 0.0001:
            logging.info("{:.2f}% of distances calculated".format((i/n_structures)*100))

    return (data_x, data_y, n_structures, n_grid_points, n_atoms)  


def create_network_one_grid_point(data_x, data_y):    
    network = Sequential()
    network.add(Dense(100, activation='relu', input_shape=((data_x.shape[-1],))))
    network.add(BatchNormalization())
    network.add(Dense(100, activation='relu'))
    network.add(BatchNormalization())
    network.add(Dense(100, activation='relu'))
    network.add(BatchNormalization())
    network.add(Dense(data_y.shape[-1], activation=LeakyReLU(alpha=0.02)))
    return network


def plot_results(prediction, data_y_test):
    # Plot predictions
    for i, prediction in enumerate(predictions):
        #fig = plt.figure(i)
        #plt.plot(predictions[i,:,0], '.', label='Prediction', alpha=0.4)
        #plt.plot(data_y_test[i,:,0], '.', label='Actual', alpha=0.4)
        #plt.xlabel("Grid point")
        #plt.ylabel("Density")
        #plt.legend()
        #plt.savefig("{}Prediction{}.png".format(plot_path, i))
        #plt.close(fig)
        header = 'Predictions, Actual'
        np.savetxt("{}result_one_output{}.dat".format(plot_path, i),
                   np.column_stack((np.array(predictions[i,:,0]), np.array(data_y_test[i,:,0]))), header=header)

        filename = 'input_one_output{}'.format(i)
        f = open(plot_path+filename, "w")
        for input in data_x_test[i]:
            f.write(str(coords))   
            f.write("\n")
        f.close()

def normalize(arr):
    return (arr-arr.mean())/np.std(arr)


#######################################
# MAIN
#######################################

# Read in x- and y-data
data_names = ['data_x', 'data_y', 'n_structures', 'n_grid_points', 'n_atoms']
exists = True
for name in data_names:
    filename_data = path_data+'{}.npy'.format(name)
    if not os.path.isfile(filename_data):
        exists = False

if exists == False or RECALCULATE == True:
    logging.info("Calculating distances from grid points to atoms")
    data_x, data_y, n_structures, n_grid_points, n_atoms = convert_cube_file(path_data) 
    np.save(path_data+'data_x.npy', data_x)
    np.save(path_data+'data_y.npy', data_y)
    np.save(path_data+'n_structures.npy', np.array(n_structures))
    np.save(path_data+'n_grid_points.npy', np.array(n_grid_points))
    np.save(path_data+'n_atoms.npy', np.array(n_atoms))
else:
    logging.info("Use existing calculations")
    data_x = np.load(path_data+'data_x.npy')    
    data_y = np.load(path_data+'data_y.npy')    
    n_structures = np.load(path_data+'n_structures.npy')    
    n_grid_points = np.load(path_data+'n_grid_points.npy')    
    n_atoms = np.load(path_data+'n_atoms.npy')

n_grid_points = 100 # ONLY FOR TESTING

print(np.amax(data_y))

#Normalize data for faster training (mean 0 and std 1)
#data_x = np.apply_along_axis(normalize, 1, data_x)
data_y_train = data_y[:90]
data_y_test = data_y[90:]
data_x_train = data_x[:90]
data_x_test = data_x[90:]


plot_path = "/home/btpl/bt702501/Dokumente/Plots_reloaded/"+parameter_set
if not os.path.isdir(plot_path):
    os.system("mkdir {}".format(plot_path))

predictions = np.zeros((len(data_x_test), n_grid_points,1))
networks = []
for i_grid_point in range(n_grid_points):#data_x.shape[1]): # ONLY FOR TESTING
    network = create_network_one_grid_point(data_x[:,i_grid_point,:], data_y[:,i_grid_point,:])    
    network.compile(optimizer=Adam(), loss='mean_absolute_error', metrics=['mse'])
    history = network.fit(data_x_train[:,i_grid_point,:], data_y_train[:,i_grid_point,:],verbose=0, validation_split=0.2, epochs=300, batch_size=16)
    results = network.evaluate(data_x_test[:,i_grid_point,:], data_y_test[:,i_grid_point,:])
    print(results)
    logging.info('Networks for {:.3f}% of grid points have been trained'.format((i_grid_point/n_grid_points)*100))
    networks.append(network)
    prediction = network.predict(data_x_test[:,i_grid_point,:])
    predictions[:,i_grid_point,:] = prediction

plot_results(predictions, data_y_test[:,:,:])

'''
training_loss = history.history['loss']
validation_loss = history.history['val_loss']
epochs = range(1,len(training_loss) +1)

# Plot whole convergence of model
plt.figure(-1)
plt.plot(epochs, training_loss, label='Training loss')
plt.plot(epochs, validation_loss, label='Validation loss')
plt.legend()
plt.savefig("Losses.png")
os.system("mv Losses.png {}".format(plot_path))
'''
'''

# Evaluate on training set
predictions = network.predict(data_x_train[:20])
for i, prediction in enumerate(predictions):
    fig = plt.figure(i+100)
    plt.plot(predictions[i], '.', label='Prediction', alpha=0.4)
    plt.plot(data_y_train[i], '.', label='Actual', alpha=0.4)
    plt.xlabel("Grid point")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("{}Training{}.png".format(plot_path, i))
    plt.close(fig)
    #filename = 'input{}'.format(i)
    #f = open(plot_path+filename, "w")
    #for input in data_x_test[i]:
    #    f.write(str(coords))   
    #    f.write("\n")
    #f.close()


print(np.amax(np.absolute(predictions[3]-predictions[7])))


# network.save("Network.h5")



# Plot second half of convergence of model
epochs_half = range(len(training_loss)//2,len(training_loss))
plt.figure(-2)
plt.plot(epochs_half, training_loss[len(training_loss)//2:], label='Training loss')
plt.plot(epochs_half, validation_loss[len(training_loss)//2:], label='Validation loss')
plt.legend()
plt.savefig("Losses_2ndHalf.png")
os.system("mv Losses_2ndHalf.png {}".format(plot_path))
'''
