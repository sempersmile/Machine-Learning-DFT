import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import csv

from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn import model_selection
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KDTree


#######################################
# Global variables and configurations
#######################################

logging.basicConfig(level=logging.INFO)

CRITICAL_MEMORY = False
ONLY_CALCULATED_GRID = True
RECALCULATE = False # Choose whether to recalculate the distances between grid-points and atoms
SUM_FINGERPRINTS = True
N_SPLITS = 5
N_GAUSSIANS = 30
DIVISOR_TOTAL_DATA = 1
PARAMETER_SET = 'Parameter_Set_77/' # 76
PATH_DATA = "/tp_leppert/bt702501/Calculated_Data/"+PARAMETER_SET
PATH_PLOT = "/home/btpl/bt702501/Dokumente/Plots_reloaded/"+PARAMETER_SET+"Epoch_Plot/"

SAVE_MODEL = False
LOAD_MODEL = False
PATH_MODEL_TO_LOAD = "/tp_leppert/bt702501/Calculated_Data/" + 'Parameter_Set_78/'

CUTOFF = 15


hyperparam_counter = 0
split_index = 1


# =====================================
# Helper functions
# =====================================


def z_score_normalization(arr_1d, mean, var):
    arr_1d = arr_1d - mean
    arr_1d = arr_1d / var
    return arr_1d 


def cutoff(r, r_c = CUTOFF, gamma=3):
    if r <= r_c:
        return 1 + gamma*(r/r_c)**(gamma+1) - (gamma+1)*(r/r_c)**gamma
    else: 
        return 0

def r_square(y_true, y_pred):
    """Custom metric for keras model"""
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


#######################################
# Different models
#######################################

class AllModels():
    def __init__(self, 
                 path_data=PATH_DATA, 
                 plot_path=PATH_PLOT, 
                 recalculate=RECALCULATE,
                 critical_memory=CRITICAL_MEMORY,
                 split_index=split_index):
        self.critical_memory = critical_memory
        self.split_index = split_index
        self.grid_spacing = 0



    # ----------------------------------------
    # Reading in Data
    # ----------------------------------------
    def get_number_of_atoms(self, 
                            path_data=PATH_DATA): 
        """
        Reads number of atoms per structure from parameter-file
        (Beware, might have to be adapted for n_atoms > 9 (more than 1 figure))
        
        Returns: number of atoms per structure
        """
        f = open(path_data+"program_parameters.dat", "r")
        params_string = f.readlines()
        self.n_atoms = int(params_string[2][-2])
        if params_string[2][-3] != ' ': 
            self.n_atoms = self.n_atoms + 10*int(params_string[2][-3])
        f.close()


    def get_folders_with_structures(self, 
                                    path_data=PATH_DATA):
        """ 
        Get the number of structures in the training data

        Returns: (list containing the folders with training data, number of structures)
        """
        self.subfolders_data = []
        self.n_structures = 0
        
        for obj in os.listdir(path_data)[:int(len(os.listdir(path_data))//self.divisor_total_data)]:
            if os.path.isdir(path_data+obj):
                self.subfolders_data.append(obj) 
                self.n_structures += 1           
        self.subfolders_data.sort()
        
        if CRITICAL_MEMORY == True:
            self.subfolders_data = self.subfolders_data[int((self.split_index-1)/N_SPLITS * len(self.subfolders_data)) : int(self.split_index/N_SPLITS * len(self.subfolders_data))]
            self.n_structures = len(self.subfolders_data)
        

    def convert_cube_file(self, 
                          path_data=PATH_DATA):
        """
        Converts the .cube-file containing the densities

        Returns: (Coordinates of grid, number of grid points)
        """
        # Read number of grid points that are actually being calculated by BTDFT from Grid_Index0000-file
        self.n_grid_points_ellip = 0
        train_folder = 'train0/'
        if PARAMETER_SET == 'Parameter_Set_23_Mixed/':
            train_folder = 'train_20_0/'
        with open(path_data+train_folder+'Grid_Index0000.dat') as f:
            self.n_grid_points_ellip = sum(1 for _ in f) - 2

        # Create array for grid with positions of grid_points 
        #(Beware, might have to be adapted for grid-dimensions that are not 2-figure numbers)
        # First, read number of grid-points from .cube-file
        f_cube = open(path_data+train_folder+"dens_orbU00.cube")
        cube_all_lines = f_cube.readlines()
        self.n_x = int(float(cube_all_lines[3][2:4]))
        self.n_y = int(float(cube_all_lines[4][2:4]))
        self.n_z = int(float(cube_all_lines[5][2:4]))
        f_cube.close()
        self.n_grid_points = self.n_x * self.n_y * self.n_z
        self.n_grid_points_cubic = self.n_grid_points
 

        # Read distances between grid-points from BTDFT_gues.conf-file 
        f_conf = open(path_data+train_folder+"/BTDFT_guess.conf", "r")
        conf_all_lines = f_conf.readlines()
        start_index = conf_all_lines[9].find("=") + 2
        end_index = conf_all_lines[9].find("b") - 1
        grid_spacing = float(conf_all_lines[9][start_index:end_index])
        self.grid_spacing = grid_spacing
        f_conf.close()
        # Create array with coordinates of each grid point
        if ONLY_CALCULATED_GRID == True:
            self.n_grid_points = self.n_grid_points_ellip
            self.grid_coords = np.zeros((self.n_grid_points, 3))
            with open(path_data+train_folder+"/Grid_Index0000.dat", "r") as f:
                all_lines = f.readlines()
                for i_line, line in enumerate(all_lines[2:]):
                    line = line.split()
                    self.grid_coords[i_line, 0] = int(line[1]) * grid_spacing
                    self.grid_coords[i_line, 1] = int(line[2]) * grid_spacing
                    self.grid_coords[i_line, 2] = int(line[3]) * grid_spacing
            self.grid_coords_ellips = self.grid_coords
        else:
            self.grid_coords = np.zeros((self.n_grid_points, 3))
            start_loop = -(self.n_x-1)/2 * grid_spacing # Assumption that all 3 dimensions in space are equal
            end_loop = (self.n_x-1)/2 * grid_spacing + grid_spacing
            point_index = 0
            for x in np.arange(start_loop, end_loop, grid_spacing):
                for y in np.arange(start_loop, end_loop, grid_spacing):
                    for z in np.arange(start_loop, end_loop, grid_spacing):
                        self.grid_coords[point_index][0] = x
                        self.grid_coords[point_index][1] = y
                        self.grid_coords[point_index][2] = z
                        point_index += 1



    def load_variable_data(self,
                           variable_names=['n_atoms', 'n_structures', 'grid_coords', 'n_grid_points', 'data_x', 'data_y'], 
                           recalculate=RECALCULATE,
                           path_data=PATH_DATA):
       
        # TODO: Rework this function (Medium priority)

        if CRITICAL_MEMORY == False and ONLY_CALCULATED_GRID == False and SUM_FINGERPRINTS == False:
            existence = True
            for name in variable_names:
                filename_data = path_data+'{}{}_{}.npy'.format(name, self.model_number, self.n_gaussians)
                if not os.path.isfile(filename_data):
                    existence = False
            if not existence or recalculate == True:
                logging.info("Calculating distances from grid points to atoms")
                self.get_number_of_atoms()
                self.get_folders_with_structures()
                self.convert_cube_file()
                self.get_advanced_fingerprints()
                variables = {
                            'n_atoms': self.n_atoms,
                            'n_structures': self.n_structures,
                            'grid_coords': self.grid_coords ,
                            'n_grid_points': self.n_grid_points,
                            'data_x': self.data_x,
                            'data_y': self.data_y,
                            }
                for name, value in variables.items():
                    if type(value) == np.ndarray:
                        np.save(path_data+name+'{}_{}.npy'.format(self.model_number, self.n_gaussians), value) 
                    else:
                        np.save(path_data+name+'{}_{}.npy'.format(self.model_number, self.n_gaussians), np.array(value))
            else:
                logging.info("Use existing calculations")
                self.data_x = np.load(path_data+'data_x{}_{}.npy'.format(self.model_number, self.n_gaussians))    
                self.data_y = np.load(path_data+'data_y{}_{}.npy'.format(self.model_number, self.n_gaussians))
                self.n_structures = np.load(path_data+'n_structures{}_{}.npy'.format(self.model_number, self.n_gaussians))    
                self.grid_coords = np.load(path_data+'grid_coords{}_{}.npy'.format(self.model_number, self.n_gaussians))
                self.n_grid_points = np.load(path_data+'n_grid_points{}_{}.npy'.format(self.model_number, self.n_gaussians))    
                self.n_atoms = np.load(path_data+'n_atoms{}_{}.npy'.format(self.model_number, self.n_gaussians))

        elif CRITICAL_MEMORY == True and ONLY_CALCULATED_GRID == False and SUM_FINGERPRINTS == False: 
            existence = True
            for name in variable_names:
                filename_data = path_data+'{}{}_{}_batch{}.npy'.format(name, self.model_number, self.n_gaussians, self.split_index)
                if not os.path.isfile(filename_data):
                    existence = False
            if not existence or recalculate == True:
                logging.info("Calculating distances from grid points to atoms")
                self.get_number_of_atoms()
                self.get_folders_with_structures()
                self.convert_cube_file()
                self.get_advanced_fingerprints()
                variables = {
                            'n_atoms': self.n_atoms,
                            'n_structures': self.n_structures,
                            'grid_coords': self.grid_coords ,
                            'n_grid_points': self.n_grid_points,
                            'data_x': self.data_x,
                            'data_y': self.data_y,
                            }
                for name, value in variables.items():
                    if type(value) == np.ndarray:
                        np.save(path_data+name+'{}_{}_batch{}.npy'.format(self.model_number, self.n_gaussians, self.split_index), value) 
                    else:
                        np.save(path_data+name+'{}_{}_batch{}.npy'.format(self.model_number, self.n_gaussians, self.split_index), np.array(value))
            else:
                logging.info("Use existing calculations")
                self.data_x = np.load(path_data+'data_x{}_{}_batch{}.npy'.format(self.model_number, self.n_gaussians, self.split_index))    
                self.data_y = np.load(path_data+'data_y{}_{}_batch{}.npy'.format(self.model_number, self.n_gaussians, self.split_index))
                self.n_structures = np.load(path_data+'n_structures{}_{}_batch{}.npy'.format(self.model_number, self.n_gaussians, self.split_index))    
                self.grid_coords = np.load(path_data+'grid_coords{}_{}_batch{}.npy'.format(self.model_number, self.n_gaussians, self.split_index))
                self.n_grid_points = np.load(path_data+'n_grid_points{}_{}_batch{}.npy'.format(self.model_number, self.n_gaussians, self.split_index))    
                self.n_atoms = np.load(path_data+'n_atoms{}_{}_batch{}.npy'.format(self.model_number, self.n_gaussians, self.split_index))
        

        elif CRITICAL_MEMORY == False and ONLY_CALCULATED_GRID == True and SUM_FINGERPRINTS == False:
            existence = True
            for name in variable_names:
                filename_data = path_data+'{}{}_only_calculated_{}.npy'.format(name, self.model_number, self.n_gaussians)
                if not os.path.isfile(filename_data):
                    existence = False
            if not existence or recalculate == True:
                logging.info("Calculating distances from grid points to atoms")
                self.get_number_of_atoms()
                self.get_folders_with_structures()
                self.convert_cube_file()
                self.get_advanced_fingerprints()
                variables = {
                            'n_atoms': self.n_atoms,
                            'n_structures': self.n_structures,
                            'grid_coords': self.grid_coords ,
                            'n_grid_points': self.n_grid_points,
                            'data_x': self.data_x,
                            'data_y': self.data_y,
                            }
                for name, value in variables.items():
                    if type(value) == np.ndarray:
                        np.save(path_data+name+'{}_only_calculated_{}.npy'.format(self.model_number, self.n_gaussians), value) 
                    else:
                        np.save(path_data+name+'{}_only_calculated_{}.npy'.format(self.model_number, self.n_gaussians), np.array(value))
            else:
                logging.info("Use existing calculations")
                self.data_x = np.load(path_data+'data_x{}_only_calculated_{}.npy'.format(self.model_number, self.n_gaussians))    
                self.data_y = np.load(path_data+'data_y{}_only_calculated_{}.npy'.format(self.model_number, self.n_gaussians))
                self.n_structures = np.load(path_data+'n_structures{}_only_calculated_{}.npy'.format(self.model_number, self.n_gaussians))    
                self.grid_coords = np.load(path_data+'grid_coords{}_only_calculated_{}.npy'.format(self.model_number, self.n_gaussians))
                self.n_grid_points = np.load(path_data+'n_grid_points{}_only_calculated_{}.npy'.format(self.model_number, self.n_gaussians))    
                self.n_atoms = np.load(path_data+'n_atoms{}_only_calculated_{}.npy'.format(self.model_number, self.n_gaussians))



        elif CRITICAL_MEMORY == True and ONLY_CALCULATED_GRID == True and SUM_FINGERPRINTS == False:
            existence = True
            for name in variable_names:
                filename_data = path_data+'{}{}_only_calculated_{}_batch{}.npy'.format(name, self.model_number, self.n_gaussians, self.split_index)
                if not os.path.isfile(filename_data):
                    existence = False
            if not existence or recalculate == True:
                logging.info("Calculating distances from grid points to atoms")
                self.get_number_of_atoms()
                self.get_folders_with_structures()
                self.convert_cube_file()
                self.get_advanced_fingerprints()
                variables = {
                            'n_atoms': self.n_atoms,
                            'n_structures': self.n_structures,
                            'grid_coords': self.grid_coords ,
                            'n_grid_points': self.n_grid_points,
                            'data_x': self.data_x,
                            'data_y': self.data_y,
                            }
                for name, value in variables.items():
                    if type(value) == np.ndarray:
                        np.save(path_data+name+'{}_only_calculated_{}_batch{}.npy'.format(self.model_number, self.n_gaussians, self.split_index), value) 
                    else:
                        np.save(path_data+name+'{}_only_calculated_{}_batch{}.npy'.format(self.model_number, self.n_gaussians, self.split_index), np.array(value))
            else:
                logging.info("Use existing calculations")
                self.data_x = np.load(path_data+'data_x{}_only_calculated_{}_batch{}.npy'.format(self.model_number, self.n_gaussians, self.split_index))    
                self.data_y = np.load(path_data+'data_y{}_only_calculated_{}_batch{}.npy'.format(self.model_number, self.n_gaussians, self.split_index))
                self.n_structures = np.load(path_data+'n_structures{}_only_calculated_{}_batch{}.npy'.format(self.model_number, self.n_gaussians, self.split_index))    
                self.grid_coords = np.load(path_data+'grid_coords{}_only_calculated_{}_batch{}.npy'.format(self.model_number, self.n_gaussians, self.split_index))
                self.n_grid_points = np.load(path_data+'n_grid_points{}_only_calculated_{}_batch{}.npy'.format(self.model_number, self.n_gaussians, self.split_index))    
                self.n_atoms = np.load(path_data+'n_atoms{}_only_calculated_{}_batch{}.npy'.format(self.model_number, self.n_gaussians, self.split_index))

          
        elif CRITICAL_MEMORY == False and ONLY_CALCULATED_GRID == True and SUM_FINGERPRINTS == True:
            existence = True
            for name in variable_names:
                filename_data = path_data+'{}{}_only_calculated_sum_{}.npy'.format(name, self.model_number, self.n_gaussians)
                if not os.path.isfile(filename_data):
                    existence = False
            if not existence or recalculate == True:
                logging.info("Calculating distances from grid points to atoms")
                self.get_number_of_atoms()
                self.get_folders_with_structures()
                self.convert_cube_file()
                self.get_advanced_fingerprints()
                variables = {
                            'n_atoms': self.n_atoms,
                            'n_structures': self.n_structures,
                            'grid_coords': self.grid_coords ,
                            'n_grid_points': self.n_grid_points,
                            'data_x': self.data_x,
                            'data_y': self.data_y,
                            }
                for name, value in variables.items():
                    if type(value) == np.ndarray:
                        np.save(path_data+name+'{}_only_calculated_sum_{}.npy'.format(self.model_number, self.n_gaussians), value) 
                    else:
                        np.save(path_data+name+'{}_only_calculated_sum_{}.npy'.format(self.model_number, self.n_gaussians), np.array(value))
            else:
                logging.info("Use existing calculations")
                self.data_x = np.load(path_data+'data_x{}_only_calculated_sum_{}.npy'.format(self.model_number, self.n_gaussians))    
                self.data_y = np.load(path_data+'data_y{}_only_calculated_sum_{}.npy'.format(self.model_number, self.n_gaussians))
                self.n_structures = np.load(path_data+'n_structures{}_only_calculated_sum_{}.npy'.format(self.model_number, self.n_gaussians))    
                self.grid_coords = np.load(path_data+'grid_coords{}_only_calculated_sum_{}.npy'.format(self.model_number, self.n_gaussians))
                self.n_grid_points = np.load(path_data+'n_grid_points{}_only_calculated_sum_{}.npy'.format(self.model_number, self.n_gaussians))    
                self.n_atoms = np.load(path_data+'n_atoms{}_only_calculated_sum_{}.npy'.format(self.model_number, self.n_gaussians))


        elif CRITICAL_MEMORY == True and ONLY_CALCULATED_GRID == True and SUM_FINGERPRINTS == True:
            existence = True
            for name in variable_names:
                filename_data = path_data+'{}{}_only_calculated_sum_{}_batch{}.npy'.format(name, self.model_number, self.n_gaussians, self.split_index)
                if not os.path.isfile(filename_data):
                    existence = False
            if not existence or recalculate == True:
                logging.info("Calculating distances from grid points to atoms")
                self.get_number_of_atoms()
                self.get_folders_with_structures()
                self.convert_cube_file()
                self.get_advanced_fingerprints()
                variables = {
                            'n_atoms': self.n_atoms,
                            'n_structures': self.n_structures,
                            'grid_coords': self.grid_coords ,
                            'n_grid_points': self.n_grid_points,
                            'data_x': self.data_x,
                            'data_y': self.data_y,
                            }
                for name, value in variables.items():
                    if type(value) == np.ndarray:
                        np.save(path_data+name+'{}_only_calculated_sum_{}_batch{}.npy'.format(self.model_number, self.n_gaussians, self.split_index), value) 
                    else:
                        np.save(path_data+name+'{}_only_calculated_sum_{}_batch{}.npy'.format(self.model_number, self.n_gaussians, self.split_index), np.array(value))
            else:
                logging.info("Use existing calculations")
                self.data_x = np.load(path_data+'data_x{}_only_calculated_sum_{}_batch{}.npy'.format(self.model_number, self.n_gaussians, self.split_index))    
                self.data_y = np.load(path_data+'data_y{}_only_calculated_sum_{}_batch{}.npy'.format(self.model_number, self.n_gaussians, self.split_index))
                self.n_structures = np.load(path_data+'n_structures{}_only_calculated_sum_{}_batch{}.npy'.format(self.model_number, self.n_gaussians, self.split_index))    
                self.grid_coords = np.load(path_data+'grid_coords{}_only_calculated_sum_{}_batch{}.npy'.format(self.model_number, self.n_gaussians, self.split_index))
                self.n_grid_points = np.load(path_data+'n_grid_points{}_only_calculated_sum_{}_batch{}.npy'.format(self.model_number, self.n_gaussians, self.split_index))    
                self.n_atoms = np.load(path_data+'n_atoms{}_only_calculated_sum_{}_batch{}.npy'.format(self.model_number, self.n_gaussians, self.split_index))

        logging.info("Mean of target data: {}".format(np.mean(self.data_y)))
        logging.info("Max of target data: {}".format(np.max(self.data_y)))
        logging.info("Min of target data: {}".format(np.min(self.data_y)))
        logging.info("Sum of target data: {}".format(np.sum(self.data_y)))
        logging.info("Number of zeros: {}".format(np.count_nonzero(self.data_y==0)))


    def train_test_split(self,
                         test_size=0.1):
        logging.info("X-shape: {} ... Y-shape: {}".format(self.data_x.shape, self.data_y.shape))
        x_train, x_test, y_train, y_test = model_selection.train_test_split(self.data_x, self.data_y, test_size=test_size, shuffle=True)
        self.data_x_train = x_train
        self.data_x_test = x_test
        self.data_y_train = y_train
        self.data_y_test = y_test



    def plot_results(self, 
                     plot_path=PATH_PLOT):
        if not os.path.isdir(plot_path):
            os.system("mkdir {}".format(plot_path))
    
        Volume_total_Bohr = 10**3
        Volume_total = (5.2918)**3
        Volume_Bohr = 1 # Taken as volume around each grid point
        Volume = 0.52918**3 # In Angstrom, for comparison with literature
        self.Volume = Volume
        # Therefore --> Density in e/A is equal to Density at each point divided by ((0.52918)**3)  

        for i, prediction in enumerate(self.predictions):
            header = 'Predictions, Actual'
            np.savetxt("{}result{}.dat".format(plot_path, i), 
               np.column_stack((np.array(self.predictions[i]/Volume), np.array(self.data_y_test[i]/Volume))),
               header=header)
            fig = plt.figure(i, figsize=[7.04,5.28])
            plt.xticks(family='cmr10')
            plt.yticks(family='cmr10')
            plt.plot(self.predictions[i]/Volume, '.', label='Prediction', alpha=0.3)
            plt.plot(self.data_y_test[i]/Volume, '.', label='Actual', alpha=0.3)
            plt.xlabel("Number of grid point", family='cmr10', size=14)
            plt.ylabel(r"Density in $\frac{e}{\AA^3}$", family='cmr10', size=14)
            plt.legend()
            plt.savefig("{}Prediction{}.png".format(plot_path, i), format='png', dpi=300)
            plt.close(fig)
            
            # Plot parity
            fig = plt.figure(i+1000, figsize=[7.04,5.28])
            plt.xticks(family='cmr10')
            plt.yticks(family='cmr10')
            plt.plot(np.linspace(min(self.data_y_test[i])/Volume, max(self.data_y_test[i])/Volume, 1000), np.linspace(min(self.data_y_test[i])/Volume, max(self.data_y_test[i])/Volume, 1000), '--', alpha=1)
            plt.plot(self.data_y_test[i]/Volume, self.predictions[i]/Volume, '.', alpha=0.4)
            plt.xlabel(r"Charge density from BTDFT in $\frac{e}{\AA^3}$", family='cmr10', size=14)
            plt.ylabel(r"Charge density predicted in $\frac{e}{\AA^3}$", family='cmr10', size=14)
            plt.savefig("{}Parity{}.png".format(plot_path, i), format='png', dpi=300)
            plt.close(fig)

            ## Plot training predictions
            #fig = plt.figure(i+2000)
            #plt.plot(self.predictions_train[i], '.', label='Prediction', alpha=0.4)
            #plt.plot(self.data_y_train[:50][i], '.', label='Actual', alpha=0.4)
            #plt.xlabel("Grid point")
            #plt.ylabel("Density")
            #plt.legend()
            #plt.savefig("{}Prediction_Training{}.png".format(plot_path, i))
            #plt.close(fig)



    def plot_loss(self,
                  plot_path=PATH_PLOT): 
        Volume = 0.52918**3 # In Angstrom, for comparison with literature
        self.Volume = Volume
        training_loss = np.array(self.history.history['loss'])/Volume
        validation_loss = np.array(self.history.history['val_loss'])/Volume
        epochs = range(1,len(training_loss) +1)

        # Plot whole convergence of model
        header = 'Epochs, Error'
        np.savetxt("{}training_error.dat".format(plot_path), 
            np.column_stack((np.array(epochs), np.array(training_loss/Volume))),header=header)
        np.savetxt("{}validation_error.dat".format(plot_path), 
        np.column_stack((np.array(epochs), np.array(validation_loss/Volume))),header=header)
        fig = plt.figure(-1, figsize=[7.04,5.28])
        plt.xticks(family='cmr10')
        plt.yticks(family='cmr10')
        plt.plot(epochs, training_loss, label='Training loss')
        plt.plot(epochs, validation_loss, label='Validation loss')
        plt.xlabel("Epoch", family='cmr10', size=14)
        plt.ylabel(r"Density in $\frac{e}{\AA^3}$", family='cmr10', size=14)
        plt.legend()
        plt.savefig("Losses.png", format='png', dpi=300)
        plt.close(fig)
        os.system("mv Losses.png {}".format(plot_path))
    
        # Plot second half of convergence of model
        epochs_half = range(len(training_loss)//2,len(training_loss))
        fig = plt.figure(-2, figsize=[7.04,5.28])
        plt.xticks(family='cmr10')
        plt.yticks(family='cmr10')
        plt.plot(epochs_half, training_loss[len(training_loss)//2:], label='Training loss')
        plt.plot(epochs_half, validation_loss[len(training_loss)//2:], label='Validation loss')
        plt.xlabel("Epoch", family='cmr10', size=14)
        plt.ylabel(r"Density in $\frac{e}{\AA^3}$", family='cmr10', size=14)
        plt.legend()
        plt.savefig("Losses_2ndHalf.png", format='png', dpi=300)
        plt.close(fig)
        os.system("mv Losses_2ndHalf.png {}".format(plot_path))
                


    def convert_cubic_to_elliptical_grid(self, data_y, convert):
        #logging.info("N Grid Points: {}".format(self.n_grid_points))
        #logging.info("N Grid Points Cubic: {}".format(self.n_grid_points_cubic))
        
        self.data_y_cubic  = np.zeros((self.n_grid_points_cubic, 4))
        self.data_y_ellipsoid = np.zeros((self.n_grid_points_ellip, 1))
        start_loop = -(self.n_x-1)/2 * self.grid_spacing # Assumption that all 3 dimensions in space are equal
        end_loop = (self.n_x-1)/2 * self.grid_spacing + self.grid_spacing - 0.00001
        point_index = 0
        for x in np.arange(start_loop, end_loop, self.grid_spacing):
            for y in np.arange(start_loop, end_loop, self.grid_spacing):
                for z in np.arange(start_loop, end_loop, self.grid_spacing):
                    self.data_y_cubic[point_index, 0] = x
                    self.data_y_cubic[point_index, 1] = y
                    self.data_y_cubic[point_index, 2] = z
                    self.data_y_cubic[point_index, 3] = data_y[point_index,0]
                    point_index += 1
        #logging.info("Shape of ellipsoid grid: {}".format(self.grid_coords_ellips.shape))
        if convert:
            counter = 0
            self.grid_indices = np.zeros((self.n_grid_points_ellip, 2), dtype=np.int64)
            tree = KDTree(self.data_y_cubic[:,:3])
            for ellips_grid_index, ellips_grid_point in enumerate(self.grid_coords_ellips):
                dist, cubic_grid_index = tree.query(ellips_grid_point.reshape(1,3), k=1)     
                self.grid_indices[counter, 0] = np.round(cubic_grid_index)
                self.grid_indices[counter, 1] = np.round(ellips_grid_index)
                self.data_y_ellipsoid[self.grid_indices[counter,1], 0] = self.data_y_cubic[self.grid_indices[counter, 0], 3]
            '''
            self.grid_indices = np.zeros((self.n_grid_points_ellip, 2))
            counter = 0
            for cubic_grid_index, cubic_grid_point in enumerate(self.data_y_cubic):
                for ellips_grid_index, ellips_grid_point in enumerate(self.grid_coords_ellips):
                    if  ((abs(cubic_grid_point[0] - ellips_grid_point[0]) < 0.01) and
                        (abs(cubic_grid_point[1] - ellips_grid_point[1]) < 0.01) and
                        (abs(cubic_grid_point[2] - ellips_grid_point[2]) < 0.01)):
                        #self.grid_indices[ellips_grid_index, 0] = ellips_grid_index
                        #self.grid_indices[ellips_grid_index, 1] = cubic_grid_index 
                        #self.data_y_ellipsoid[ellips_grid_index] = cubic_grid_point[3]
                        self.grid_indices[counter, 0] = cubic_grid_index
                        self.grid_indices[counter, 1] = ellips_grid_index
                        self.data_y_ellipsoid[int(self.grid_indices[counter,1]), 0] = self.data_y_cubic[int(self.grid_indices[counter, 0]), 3]
                        counter += 1
                        break
                if abs(int((cubic_grid_index/self.n_grid_points_cubic)*100) - ((cubic_grid_index/self.n_grid_points_cubic)*100)) < 0.001:
                    logging.info("{:.2f}% of cubic grid points mapped to elliptical grid".format(((cubic_grid_index/self.n_grid_points_cubic)*100 + 1)))
            '''
            logging.info("Mapping of cubic grid points to elliptical grid finished")
        else:
            for indices in self.grid_indices:
                logging.info("{}".format(self.data_y_cubic[indices[0], 3]))
                self.data_y_ellipsoid[indices[1], 0] = self.data_y_cubic[indices[0], 3]
        
        logging.info("Number of zeros (in converting function): {}".format(np.count_nonzero(self.data_y_ellipsoid==0)))
        
        return self.data_y_ellipsoid




# TODO: Set up Model1 and Model2
class Model1(AllModels):
    """
    Most simple neuronal network.
    
    Input:  Flattend coordinates of all atoms in the structure (i.e. one input neuron for each dimension and each atom)
    Output: The density value at each grid point

    Comments: Models seems to have a big problem with underfitting
    """
    def __init__(self,
                 path_data=PATH_DATA,
                 plot_path=PATH_PLOT,
                 recalculate=RECALCULATE):
        super().__init__(path_data,
                         plot_path,
                         recalculate)
        self.model_number = 1
    

class Model2():
    """
    First try in fingerprinting input. 
    For each grid point, the euclidean distances of the grid point to each atom are
    calculated and then summed up. For each grid point this sum is given to the neuronal network as an input.

    Input:  For all grid points: The sum of the distances of the grid point to each atom
    Output: The density value at each grid point

    Comments:
    """
    def __init__(self, 
                 path_data,
                 plot_path, 
                 recalculate):
        super().__init__(path_data=PATH_DATA, plot_path=PLOT_PATH, recalculate=RECALCULATE)
        self.model_number = 1



class Model3(AllModels):
    """
    A bit more sophisticated fingerprinting compared to NeuronalNetwork2().   
    For each grid point, the euclidean distance of the grid point to each atom is calculated. These distances
    are then fed into the neuronal network. This means that there is an individual neuronal network for each
    grid point.

    Input:  The distances of the grid point to each atom
    Output: The density value at the given grid point  

    Comments: 
    """   
    def __init__(self, 
                 path_data=PATH_DATA,
                 plot_path=PATH_PLOT,
                 recalculate=RECALCULATE):
        super().__init__(path_data, 
                         plot_path,
                         recalculate)
        self.model_number = 3

    # ----------------------------------------
    # Data Preprocessing
    # ----------------------------------------
    
    def get_fingerprints(self,
                         path_data=PATH_DATA):
        structure_coords = np.zeros((self.n_structures, self.n_atoms, 3))
        self.data_x = np.zeros((self.n_structures, self.n_grid_points, self.n_atoms))
        self.data_y = np.zeros((self.n_structures, self.n_grid_points, 1))

        for i, directory in enumerate(self.subfolders_data):
            # Read densities at grid-points from .cube-file and coordinates from .npy-file
            f_read = open(path_data+directory+"/dens_orbU00.cube", "r")
            all_lines = f_read.readlines()
            f_read.close()
            all_lines_split = [line.split() for line in all_lines[6+self.n_atoms:]]
            all_lines_flattened = [y.replace('\n','') for x in all_lines_split for y in x]
            data_single_structure = np.array((all_lines_flattened,)) 
            data_single_structure = data_single_structure[data_single_structure!='']
            data_single_structure = data_single_structure.astype(np.float32)
            # WARNING: Note that the order of the points might be odd 
            #data_single_structure = data_single_structure[data_single_structure!=0.0] 
            self.data_y[i,:,:] = data_single_structure.reshape(-1,1)

            structure_coords[i,:,:] = np.load(path_data+directory+'/coordinates.npy')
            
            # Calculate distances
            for i_grid, grid_point in enumerate(self.grid_coords):
                for i_atom, atom_coords in enumerate(structure_coords[i,:,:]):
                    self.data_x[i,i_grid,i_atom] = np.linalg.norm(atom_coords-grid_point)
            
            if abs(int((i/self.n_structures)*100) - (i/self.n_structures)*100) < 0.00001:
                logging.info("{:.2f}% of distances calculated".format((i/self.n_structures)*100))

    def train_test_split(self):
        self.data_y_train = self.data_y[:400]
        self.data_y_test = self.data_y[450:]
        self.data_x_train = self.data_x[:400]
        self.data_x_test = self.data_x[450:]

    
    def create_network_one_grid_point(self):    
        network = Sequential()
        network.add(BatchNormalization())
        network.add(Dense(150, activation='relu', input_shape=((self.data_x.shape[-1],))))
        network.add(BatchNormalization())
        network.add(Dense(150, activation='tanh'))
        network.add(BatchNormalization())
        network.add(Dense(150, activation='relu'))
        network.add(BatchNormalization())
        network.add(Dense(self.data_y.shape[-1]))
        
        network.compile(optimizer=Adam(), loss='mean_absolute_error', metrics=['mse'])
        
        return network


    def train_networks_and_predict(self,
                                   plot_path=PATH_PLOT):
        self.predictions = np.zeros((len(self.data_x_test), self.n_grid_points,1))
        self.networks = []
        for i_grid_point in range(self.data_x.shape[1]):#n_grid_points):#self.data_x.shape[1]): # ONLY FOR TESTING
            network = self.create_network_one_grid_point()    
            history = network.fit(self.data_x_train[:,i_grid_point,:], self.data_y_train[:,i_grid_point,:],
                                  verbose=0, validation_split=0.2, epochs=500, batch_size=8)
            results = network.evaluate(self.data_x_test[:,i_grid_point,:], self.data_y_test[:,i_grid_point,:])
            print(results)
            logging.info('Networks for {:.3f}% of grid points have been trained'.format((i_grid_point/self.n_grid_points)*100))
            self.networks.append(network)
            prediction = network.predict(self.data_x_test[:,i_grid_point,:])
            self.predictions[:,i_grid_point,:] = prediction

        # Reshape into original shape for plotting 
        self.predictions = self.predictions.reshape(self.orig_shape_y_test)[:,:,0]
        self.data_y_test = self.data_y_test.reshape(self.orig_shape_y_test)[:,:,0]


class Model4(AllModels):
    """
    Based on Model3, but using the same network for any grid point
    """ 
    def __init__(self,
                 path_data=PATH_DATA,
                 plot_path=PATH_PLOT,
                 recalculate=RECALCULATE,
                 divisor_total_data=DIVISOR_TOTAL_DATA):
        super().__init__(path_data=PATH_DATA,
                         plot_path=PATH_PLOT,
                         recalculate=RECALCULATE)
        self.model_number = 4
        self.n_gaussians = N_GAUSSIANS
        self.divisor_total_data = divisor_total_data

    # ----------------------------------------
    # Data Preprocessing
    # ----------------------------------------
    
    def get_fingerprints(self,
                         path_data=PATH_DATA):
        structure_coords = np.zeros((self.n_structures, self.n_atoms, 3))
        self.data_x = np.zeros((self.n_structures, self.n_grid_points, self.n_atoms))
        self.data_y = np.zeros((self.n_structures, self.n_grid_points, 1))

        for i, directory in enumerate(self.subfolders_data):
            # Read densities at grid-points from .cube-file and coordinates from .npy-file
            f_read = open(path_data+directory+"/dens_orbU00.cube", "r")
            all_lines = f_read.readlines()
            f_read.close()
            all_lines_split = [line.split('    ') for line in all_lines[6+self.n_atoms:]]
            all_lines_flattened = [y.replace('\n','') for x in all_lines_split for y in x]
            data_single_structure = np.array((all_lines_flattened,)) 
            data_single_structure = data_single_structure[data_single_structure!='']
            data_single_structure = data_single_structure.astype(np.float64)
            # WARNING: Note that the order of the points might be odd 
            #data_single_structure = data_single_structure[data_single_structure!=0.0] 
            self.data_y[i,:,:] = data_single_structure.reshape(-1,1)

            structure_coords[i,:,:] = np.load(path_data+directory+'/coordinates.npy')
            
            # Calculate distances
            for i_grid, grid_point in enumerate(self.grid_coords):
                for i_atom, atom_coords in enumerate(structure_coords[i,:,:]):
                    self.data_x[i,i_grid,i_atom] = np.linalg.norm(atom_coords-grid_point)
            
            if abs(int((i/self.n_structures)*100) - (i/self.n_structures)*100) < 0.0001:
                logging.info("{:.2f}% of distances calculated".format((i/self.n_structures)*100 + 1))

    
    def get_advanced_fingerprints(self,
                                  n_gaussians=N_GAUSSIANS,
                                  path_data=PATH_DATA):
        # For implementation of vectorial and tensorial features
        structure_coords = np.zeros((self.n_structures, self.n_atoms, 3))
        if SUM_FINGERPRINTS:
            self.data_x = np.zeros((self.n_structures, self.n_grid_points, self.n_gaussians*5))
        else:
            self.data_x = np.zeros((self.n_structures, self.n_grid_points, self.n_atoms*self.n_gaussians*5))
        self.data_y = np.zeros((self.n_structures, self.n_grid_points, 1))
        data_y_cubic_temp = np.zeros((self.n_structures, self.n_grid_points_cubic, 1))

        for i, directory in enumerate(self.subfolders_data):
            logging.info("Structure {}".format(i)) 
            # Read densities at grid-points from .cube-file and coordinates from .npy-file
            f_read = open(path_data+directory+"/dens_orbU00.cube", "r")
            all_lines = f_read.readlines()
            f_read.close()
            all_lines_split = [line.split() for line in all_lines[6+self.n_atoms:]]
            all_lines_flattened = [y.replace('\n','') for x in all_lines_split for y in x]
            data_single_structure = np.array((all_lines_flattened,)) 
            data_single_structure = data_single_structure[data_single_structure!='']
            data_single_structure = data_single_structure.astype(np.float32)
            # WARNING: Note that the order of the points might be odd 
            # data_single_structure = data_single_structure[data_single_structure != 0.0] 
            if  ONLY_CALCULATED_GRID == True:
                if i >= 0:
                    convert = True
                else:
                    convert = False
                data_y_cubic_temp[i,:,:] = data_single_structure.reshape(-1,1)
                self.data_y[i,:,:] = self.convert_cubic_to_elliptical_grid(data_y_cubic_temp[i,:,:], convert)
            else:
                self.data_y[i] = data_single_structure.reshape(-1,1)

            logging.info("Shape data_single_structure: {}".format(data_single_structure.shape))
            logging.info("Shape data_x: {}".format(self.data_x.shape))
            logging.info("Shape data_y: {}".format(self.data_y.shape))
            logging.info("Number of zeros (in fingerprinting function): {}".format(np.count_nonzero(self.data_y[i]==0)))
            
            #if abs(int((i/self.n_structures)*100) - (i/self.n_structures)*100) < 0.01:
            logging.info("{:.2f}% of distances calculated".format((i/self.n_structures)*100))

            structure_coords[i,:,:] = np.load(path_data+directory+'/coordinates.npy')
            # Calculate distances
            for i_grid, grid_point in enumerate(self.grid_coords):
                for i_gaussian, sigma in enumerate(np.logspace(-0.8, 1.1, num=self.n_gaussians, base=10.0)):
                    C_norm = (2*pi)**(-3/2)*sigma**(-3)
                    # Creating rotationally invariant fingerprints --> Chandrasekaran et al.
                    for i_atom, atom_coords in enumerate(structure_coords[i,:,:]):
                        r_2 =  (np.linalg.norm(atom_coords-grid_point))**2 
                        #if np.sqrt(r_2) > CUTOFF:
                        #    break
                        exp_func = np.exp(-r_2/(2*sigma**2))         

                        # Scalar fingerprint
                        gaussian = C_norm * exp_func
                        S = gaussian #* cutoff(np.sqrt(r_2))
                        # Vectorial fingerprint
                        V_dim = np.zeros((3))
                        V = np.zeros((1))
                        for dim in [0,1,2]:
                            r_dim = abs(atom_coords[dim] - grid_point[dim])
                            gaussian = C_norm * r_dim/(2*sigma**2) * exp_func
                            V_dim[dim] = gaussian #* cutoff(np.sqrt(r_2))
                        V = np.linalg.norm(V_dim)
                        # Tensorial fingerprints
                        T_dim = np.zeros((3,3))                        
                        T = np.zeros((3))
                        for dim1 in [0,1,2]:
                            for dim2 in [0,1,2]:
                                r_dim1 = abs(atom_coords[dim1] - grid_point[dim1])
                                r_dim2 = abs(atom_coords[dim2] - grid_point[dim2])
                                gaussian = C_norm * r_dim1 * r_dim2 /(4*sigma**4) * exp_func
                                T_dim[dim1,dim2] = gaussian #* cutoff(np.sqrt(r_2))
                        T[0] = T_dim[0,0]**2 + T_dim[1,1]**2 + T_dim[2,2]**2
                        T[1] = (T_dim[0][0]*T_dim[1][1] + T_dim[1][1]*T_dim[2][2] + T_dim[0][0]*T_dim[2][2]
                              - (T_dim[0][1])**2 - (T_dim[1][2])**2 - (T_dim[2][0])**2)
                        T[2] = (T_dim[0][0]*T_dim[1][1]*T_dim[2][2]
                              + T_dim[0][1]*T_dim[1][2]*T_dim[2][0]
                              + T_dim[0][2]*T_dim[1][0]*T_dim[2][1]
                              - T_dim[0][2]*T_dim[1][1]*T_dim[2][0]
                              - T_dim[0][0]*T_dim[1][2]*T_dim[2][1]
                              - T_dim[0][1]*T_dim[1][0]*T_dim[2][2]) 
                        if SUM_FINGERPRINTS:
                            self.data_x[i, i_grid, 5*i_gaussian] += S
                            self.data_x[i, i_grid, 5*i_gaussian+1] += V
                            self.data_x[i, i_grid, 5*i_gaussian+2] += T[0]
                            self.data_x[i, i_grid, 5*i_gaussian+3] += T[1]
                            self.data_x[i, i_grid, 5*i_gaussian+4] += T[2]
                        else:
                            self.data_x[i, i_grid, self.n_gaussians*5*i_atom+5*i_gaussian] = S
                            self.data_x[i, i_grid, self.n_gaussians*5*i_atom+5*i_gaussian+1] = V
                            self.data_x[i, i_grid, self.n_gaussians*5*i_atom+5*i_gaussian+2] = T[0]
                            self.data_x[i, i_grid, self.n_gaussians*5*i_atom+5*i_gaussian+3] = T[1]
                            self.data_x[i, i_grid, self.n_gaussians*5*i_atom+5*i_gaussian+4] = T[2]


        logging.info("Original shape of input data: {}".format(self.data_x.shape))             
        self.data_x = self.data_x.reshape(self.data_x.shape[0], self.data_x.shape[1], -1)
        logging.info("Shape of input data after reshaping: {}".format(self.data_x.shape))

    
    def prepare_data_for_network(self):
        # Save original shape of output data
        self.orig_shape_y_test = self.data_y_test.shape

        # Normalizing columns for each sample point --> Separately for training and test data
        for i_sample in range(self.data_x_train.shape[0]):
            for i_col in range(self.data_x_train.shape[2]):
                arr = self.data_x_train[i_sample, :, i_col]
                mean = np.mean(arr, axis=0)
                var = np.std(arr, axis=0)
                self.data_x_train[i_sample, :, i_col] = z_score_normalization(arr, mean, var)        
        for i_sample in range(self.data_x_test.shape[0]):
            for i_col in range(self.data_x_test.shape[2]):
                arr = self.data_x_test[i_sample, :, i_col]
                mean = np.mean(arr, axis=0)
                var = np.std(arr, axis=0)
                self.data_x_test[i_sample, :, i_col] = z_score_normalization(arr, mean, var)        


        # Reshape training data
        self.data_x_train = self.data_x_train.reshape(self.data_x_train.shape[0]*self.data_x_train.shape[1], -1)
        self.data_y_train = self.data_y_train.reshape(self.data_y_train.shape[0]*self.data_y_train.shape[1], -1) 
        self.data_x_test = self.data_x_test.reshape(self.data_x_test.shape[0]*self.data_x_test.shape[1], -1)
        self.data_y_test = self.data_y_test.reshape(self.data_y_test.shape[0]*self.data_y_test.shape[1], -1)

        ### Global normalizing for all grid points
        #x_all = np.vstack((self.data_x_train, self.data_x_test))
        #logging.info("Min: {}".format(x_all.min()))
        #logging.info("Mean: {}".format(x_all.mean()))
        #mean = np.mean(x_all, axis=0)
        #var = np.std(x_all, axis=0)
        
        #for i in range(self.data_x_train.shape[1]):
        #    arr = self.data_x_train[:, i].copy()
        #    self.data_x_train[:, i] = z_score_normalization(arr, mean[i], var[i]) 
        #for i in range(self.data_x_test.shape[1]):
        #    arr = self.data_x_test[:, i].copy()
        #    self.data_x_test[:, i] = z_score_normalization(arr, mean[i], var[i])
         

        logging.info("Shape of data_x_train: {}".format(self.data_x_train.shape))             
        logging.info("Shape of data_y_train: {}".format(self.data_y_train.shape))             
        logging.info("Shape of data_x_test: {}".format(self.data_x_test.shape))             
        logging.info("Shape of data_y_test: {}".format(self.data_y_test.shape))             


    def create_network(self,
                       track_progress_hyperparam_tuning=0,
                       activation='relu',
                           n=700,
                       dropout=0.2,
                       learning_rate=0.0002,
                       amsgrad=False):
        '''
        Comments on choosing hyperparame:
        elu activation function has shown significantly better results during hyperparameter optimization
        compared to relu and tanh
        '''
        
        self.activation = activation
        self.n = n
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.amsgrad = amsgrad
        
    
        self.optimizer = Adam(amsgrad=amsgrad, learning_rate=learning_rate) # Maybe try out different learning rates
        
        network = Sequential()
        network.add(Dense(n, activation='relu', input_shape=((self.data_x.shape[-1],))))
        network.add(Dropout(dropout))
        #network.add(Dense(n, activation='tanh'))
        #network.add(Dropout(dropout))
        network.add(Dense(n, activation='relu'))
        network.add(Dropout(dropout))
        #network.add(Dense(n, activation='elu'))
        #network.add(Dropout(dropout))
        #network.add(Dense(n, activation='relu'))
        #network.add(Dropout(dropout))
        #network.add(Dense(n, activation='tanh'))
        #network.add(Dropout(dropout))
        #network.add(Dense(n, activation='elu'))
        #network.add(Dropout(dropout))
        #network.add(Dense(n, activation='tanh'))
        #network.add(Dropout(dropout)) 
        #network.add(Dense(n, activation='relu'))
        #network.add(Dropout(dropout))
        #network.add(Dense(n, activation='relu'))
        #network.add(Dropout(dropout))
        #network.add(Dense(n, activation='relu'))
        #network.add(Dropout(dropout))
        #network.add(Dense(n, activation='relu'))
        #network.add(Dropout(dropout))
        network.add(Dense(n, activation='relu'))
        network.add(Dropout(dropout))
        network.add(Dense(self.data_y.shape[-1]))

        network.compile(optimizer=self.optimizer, loss='mean_squared_error', metrics=['mae', r_square])

        if track_progress_hyperparam_tuning == 1:
            global hyperparam_counter
            hyperparam_counter += 1
            # TODO: Rework hyperparam_counter
            print("{} of {} possible points in the parameter space have been tried".format(hyperparam_counter, self.n_params_grid))

        return network


    def train_network(self,
                      network,
                      epochs=1000):
        if not LOAD_MODEL:
            x_train, x_val, y_train, y_val = model_selection.train_test_split(self.data_x_train, self.data_y_train, test_size=0.2, shuffle=True)
            
            epochs = epochs
            batch_size = 2048
            self.history = network.fit(x_train,
                                       y_train,
                                       validation_data=(x_val, y_val),
                                       epochs=epochs,
                                       batch_size=batch_size,
                                       shuffle=True)
            self.epochs = epochs
            self.batch_size = batch_size        
            self.network = network            
            if SAVE_MODEL:
               network.save(PATH_DATA+"model.h5")

        else: 
            network = load_model(PATH_MODEL_TO_LOAD+"model.h5")
            self.network = network

        return network


    def predict(self,
                network):
        logging.info("Shape of data_x_test just before prediction: {}".format(self.data_x_test.shape))
        self.predictions = network.predict(self.data_x_test)
        self.results = network.evaluate(self.data_x_test, self.data_y_test)
        print(self.results)
        self.rmse = np.sqrt(self.results[0])*((1/0.52918)**3) 
        print("RMSE", self.rmse) 
        # Reshape into original shape for plotting 
        self.predictions = self.predictions.reshape(self.orig_shape_y_test)[:,:,0]
        self.data_y_test = self.data_y_test.reshape(self.orig_shape_y_test)[:,:,0]
        #logging.info(np.min(self.data_y_test))
    
        self.network = network

        
    def save_results(self,
                        plot_path=PATH_PLOT):
        with open(plot_path+"Results.dat", "a") as file_results:
            file_results.write("\n=========================================\n")
            if not LOAD_MODEL:
                file_results.write("Parameters:\n")
                file_results.write("Load Model: {}\n".format(LOAD_MODEL))
                file_results.write("Only calculated Grid: {}\n".format(ONLY_CALCULATED_GRID))
                file_results.write("Sum fingerprints: {}\n".format(SUM_FINGERPRINTS))
                file_results.write("N Structures: {}\n".format(self.n_structures))
                file_results.write("N Grid points: {}\n".format(self.n_grid_points)) 
                file_results.write("N Gaussians: {}\n".format(self.n_gaussians))
                file_results.write("Epochs: {}\n".format(self.epochs))
                file_results.write("Batch Size: {}\n".format(self.batch_size))
                file_results.write("Optimizer: {}\n".format(self.optimizer))
                file_results.write("Amsgrad: {}\n".format(self.amsgrad))
                file_results.write("Learning Rate: {}\n".format(self.learning_rate))
                file_results.write("Activation: {}\n".format(self.activation))
                file_results.write("N Neurons: {}\n".format(self.n))
                file_results.write("Dropout: {}\n".format(self.dropout)) 
                file_results.write("Results (MSE,MAE): ({},{})\n".format(self.results[0], self.results[1])) 
                file_results.write("RMSE (e/A^3): {}\n".format(self.rmse)) 
            else: 
                file_results.write("Load Model: {}\n".format(PATH_MODEL_TO_LOAD))
                file_results.write("Only calculated Grid: {}\n".format(ONLY_CALCULATED_GRID))
                file_results.write("Sum fingerprints: {}\n".format(SUM_FINGERPRINTS))
                file_results.write("Results (MSE,MAE): ({},{})\n".format(self.results[0], self.results[1])) 
                file_results.write("RMSE (e/A^3): {}\n".format(self.rmse)) 
                

        self.network.summary()

    def hyperparameter_tuning(self,
                              plot_path=PATH_PLOT):
        model = KerasRegressor(build_fn=self.create_network, verbose=0)
        
        # Define grid search parameters
        track_progress_hyperparam_tuning = [1]
        batch_size = [5000]
        epochs = [500]
        activation = ['relu']
        n = [100, 200, 300, 400, 500, 600, 700, 800, 900, 100]
        dropout = [0.1]
        param_grid = dict(track_progress_hyperparam_tuning=track_progress_hyperparam_tuning,
                          batch_size=batch_size,
                          epochs=epochs,
                          activation=activation,
                          n=n,
                          dropout=dropout)

        cv = 3
        self.n_params_grid = 1
        for key, value in param_grid.items():
            if key != 'track_progress_hyperparam_tuning':
                self.n_params_grid *= len(value) 
        self.n_params_grid *= cv
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=cv, verbose=1)

        grid_result = grid.fit(self.data_x_train, self.data_y_train)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        f = open(plot_path+"hyperparameter_tuning_results.dat","a")
        file_results.write("\n=========================================\n")
        f.write("Best: %f using %s\n" % (grid_result.best_score_, grid_result.best_params_))
        for mean, stdev, param in zip(means, stds, params):
            print("Mean MSE: %f (std: %f) with: %r" % (mean, stdev, param)) 
            f.write("Mean MSE: %f (std: %f) with: %r\n" % (mean, stdev, param))
        f.close()


    def train_model_svr(self):
        x_train, x_val, y_train, y_val = model_selection.train_test_split(self.data_x_train, self.data_y_train, test_size=0.2, shuffle=True)
        self.clf = svm.SVR(epsilon=.0012)
        print(x_train.shape)        
        self.clf.fit(x_train[0], y_train[0])
        
    
    def predict_svr(self):
        print(self.data_x_test.shape)
        self.predictions = self.clf.predict(self.data_x_test[0])


    def plot_results_svr(self, 
                     plot_path=PATH_PLOT):
        if not os.path.isdir(plot_path):
            os.system("mkdir {}".format(plot_path))

        fig = plt.figure(0)
        plt.plot(self.predictions, '.', label='Prediction', alpha=0.4)
        plt.plot(self.data_y_test[0], '.', label='Actual', alpha=0.4)
        plt.xlabel("Grid point")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig("{}Prediction_SVR{}.png".format(plot_path, 0))
        plt.close(fig)
            



class Model5(AllModels):
    """
    First tries using support vector regression
    """ 
    def __init__(self,
                 path_data=PATH_DATA,
                 plot_path=PATH_PLOT,
                 recalculate=RECALCULATE,
                 divisor_total_data=DIVISOR_TOTAL_DATA):
        super().__init__(path_data=PATH_DATA,
                         plot_path=PATH_PLOT,
                         recalculate=RECALCULATE)
        self.model_number = 5
        self.n_gaussians = N_GAUSSIANS
        self.divisor_total_data = divisor_total_data



#######################################
# MAIN
#######################################

def main():
    logging.info("Plotting data from folder {}".format(PARAMETER_SET))

    '''
    model = Model4() 
    model.load_variable_data()
    model.train_test_split()
    model.train_model()
    model.predict()
    model.plot_results_svr()

             

    
    # Convergence N Epochs
    for epoch in range(1, 16, 3):
        print("Epoch:", epoch)
        model = Model4()
        model.load_variable_data()
        model.train_test_split()
        model.prepare_data_for_network()
        network = model.create_network()
        network = model.train_network(network, epochs=50*epoch)
        model.predict(network)
        model.save_results()
        #model.plot_results()
        #model.plot_loss()
    
      
    # Convergence N Neurons
    for neurons in range(250, 2250, 250):
        print("Neurons:", neurons)
        model = Model4()
        model.load_variable_data()
        model.train_test_split()
        model.prepare_data_for_network()
        network = model.create_network(n=neurons)
        network = model.train_network(network, epochs=500)
        model.predict(network)
        model.save_results()
        #model.plot_results()
        #model.plot_loss()
    

     
    # Convergence N Sample Structures
    for divisor in [10, 5, 3, 2]:
        print("Divisor:", divisor)
        model = Model4(divisor_total_data=divisor)
        model.load_variable_data()
        model.train_test_split()
        model.prepare_data_for_network()
        network = model.create_network()
        network = model.train_network(network)
        model.predict(network)
        model.save_results()
        model.plot_results()
        model.plot_loss()
    '''
    
    if CRITICAL_MEMORY:
        model = Model4()
        for i in range(1, N_SPLITS):
            model.split_index = i 
            logging.info("Training model on batch {}".format(model.split_index))
            model.load_variable_data()
            model.train_test_split()
            model.prepare_data_for_network()
            if i == 1:
                network = model.create_network()
            network = model.train_network(network)
            model.predict(network) 
            model.save_results()
            model.plot_results()
            model.plot_loss()
        
    else:
        model = Model4()
        model.load_variable_data()
        model.train_test_split()
        model.prepare_data_for_network()
        network = model.create_network()
        network = model.train_network(network)
        model.predict(network)
        model.save_results()
        model.plot_loss()        
        model.plot_results()
    
    
if __name__ == "__main__":
    main()
