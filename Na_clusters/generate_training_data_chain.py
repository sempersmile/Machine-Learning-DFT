import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

#######################################
# Setting global parameters
#######################################

RANDOM = True

N_STRUCTURES = int(input("Number of structures to generate? "))# Total number of Na-structures
N_ATOMS = 2 # Number of Na-atoms per Na-structure
PATH = "/tp_leppert/bt702501/Calculated_Data/Parameter_Set_25/"

BOND_LENGTH_MIN_ANGSTROM = 2.5# Param20: 2.5 ,Param19: 2.0 # Minimum bond length in units of Angstrom (set to radius of Na), used vdW-radius
BOND_LENGTH_MAX_ANGSTROM = 2.6# Param20: 2.6 ,Param19: 2.1 # Maximum bond length in units of Angstrom
BOND_LENGTH_MIN = BOND_LENGTH_MIN_ANGSTROM * 1.889725989
BOND_LENGTH_MAX = BOND_LENGTH_MAX_ANGSTROM * 1.889725989
BOX_SIZE = 9.0 # TODO: Orig: 9.0# Box size in Bohr (--> Box in range -BOX_SIZE to +BOX_SIZE)
DIMENSION = 3 # Specify dimension of problem

#######################################
# Functions
#######################################

def get_random_structure(n_atoms=N_ATOMS, bond_length_min=BOND_LENGTH_MIN,
                       bond_length_max=BOND_LENGTH_MAX,dim=DIMENSION,box_size=BOX_SIZE): 
    coords=np.zeros((n_atoms,dim))
    coords[0,0] = np.random.uniform(-8,-7,(1)) # Param19: -8, -7
    coords[0,1:] = np.random.uniform(-3,3,(2)) # Param19: -2, 2
    in_box = False
    while in_box == False:
        in_box = True
        if dim == 1:
            pass
            # Advance in x-direction
            for counter, atom in enumerate(coords[1:,:], 1):
                coords[counter,0] = coords[counter-1,0] + np.random.uniform(bond_length_min, bond_length_max)
                assert np.linalg.norm(coords[counter,:]-coords[counter-1,:]) >= bond_length_min
                assert np.linalg.norm(coords[counter,:]-coords[counter-1,:]) <= bond_length_max
        elif dim == 2 or dim == 3:
            # Advance in x-direction
            if RANDOM:
                for counter, atoms in enumerate(coords[1:,:], 1):
                    x = np.random.uniform(4.9, 5.1, (1))[0] # x-coordinate
                    yz = np.random.uniform(-0.5, 0.5, (dim-1)) # Param19: -0.5, 0.5
                    scaling_factor = np.linalg.norm(np.array([x, yz[0], yz[1]])) / np.random.uniform(bond_length_min, bond_length_max, (1))[0]
                    coords[counter, 0] = coords[counter-1, 0] + x/scaling_factor
                    coords[counter, 1:] = coords[counter-1, 1:] + yz/scaling_factor
                    assert np.linalg.norm(coords[counter,:dim]-coords[counter-1,:dim]) >= bond_length_min
                    assert np.linalg.norm(coords[counter,:dim]-coords[counter-1,:dim]) <= bond_length_max
                    if np.max(np.absolute(coords[counter,:])) > BOX_SIZE:
                        in_box = False
            else:
                coords[1,:] = coords[0,:] + np.array([0, 2.89, 6.14])
                coords[2,:] = coords[1,:] - 2*np.array([0, 0, 6.14])
                coords[3,:] = coords[2,:] + np.array([0, 2.89, 6.14])

        logging.info(coords)   
    return coords


def get_random_samples(n_structures=N_STRUCTURES, n_atoms=N_ATOMS, dim=DIMENSION,
                      bond_length_min=BOND_LENGTH_MIN, bond_length_max=BOND_LENGTH_MAX):
    ''' 
    Give arguments in the following way: example_arg=example_value
    
    Following arguments can be given:
    - n_structures (Number of structures in the sample)
    - n_atoms (Number of atoms per structure)
    - dim (Local dimension, generally 3)
    - bond_length_min (Minimum distance between two atoms, in Angstrom)
    - bond_length_max (Maximum distance between two atoms, in Angstrom)
    '''
    structures = np.array([get_random_structure() for i in range(n_structures)])
    return structures

def modify_guess_file(coords, n_atoms=N_ATOMS, inputfile="BTDFT_guess_orig.conf", outputfile="BTDFT_guess.conf"):
    # Read from old file
    readfile = open(inputfile, "r")
    all_lines = readfile.readlines()
    readfile.close()
    
    del all_lines[41:]
    for i in range(n_atoms):
        coords_atom = "{:10s}\t{:10s}\t{:10s}\n".format(
                        str((round(coords[i][0],5))), str(round(coords[i][1],5)), str(round(coords[i][2],5)))
        all_lines.append(coords_atom) 
    all_lines.append("end atom_coord")
    # Write to new file
    writefile = open(outputfile, "w")
    writefile.writelines(all_lines)
    writefile.close()

def create_folders_all_structures(path, all_structures):
    for i, structure in enumerate(all_structures):
        logging.info("\n*************************\nStructure Number {}\n*************************".format(i+1))
        logging.debug("Modifying BTDFT_guess.conf-file...")
        modify_guess_file(structure)
        logging.debug("BTDFT_guess.conf-file has been modifyed")  
        logging.debug("Creating initial guess...")
        os.system("./BTDFT_guess.tp4local") 
        logging.debug("Initial guess has been created") # Maybe redirect output directly to trainfolder?
        logging.info("Calculating ground state... this might take a while...")
        os.system("./BTDFT_gs.tp4local")
        if os.path.isfile("ground_state.ace"):
            logging.info("Calculation of ground state successful")
            os.system("./ace2human.tp4local -i ground_state.ace -q dens -t cube")
            logging.debug("Creating new folder...")
            folder_number = i
            while os.path.isdir(path+"train{}".format(folder_number)):
                folder_number += 1
            os.system("mkdir {}train{}".format(path,folder_number))
            logging.debug("Folder has been created...")
            np.save(path+'train{}/coordinates.npy'.format(folder_number), structure)
            os.system("mv *.pdb *.xyz *.dat *.stat *cube *.out *.0000 {}train{}".format(path,folder_number))
            os.system("cp *s.conf {}train{}".format(path,folder_number)) # Save -config files too
            os.system("rm *.ace")
            if os.path.isfile("BTDFT.ERROR"):
                os.system("mv BTDFT.ERROR {}train{}".format(path,folder_number))
        else:
            logging.info("Calculation of ground state NOT SUCCESSFUL")
            os.system("rm *.pdb *.xyz *.dat *.stat *cube *.out *.0000")
            os.system("rm *.ace")
            if os.path.isfile("BTDFT.ERROR"):
                os.system("rm BTDFT.ERROR")

#######################################
# Main
#######################################

logging.info("\n========================================\nSTART OF PROGRAM\n========================================")

logging.debug("Sampling structures...")
all_structures = get_random_samples()
logging.debug("All structures have been sampled")

path = PATH

if os.path.isdir(path) == False:
    os.system("mkdir {}".format(path))

file = open(path+"program_parameters.dat", "w+")
file.write("Parameters for generating training structures\n\n")
file.write("Number of atoms per structure: {}\n".format(N_ATOMS))
file.write("Minimum bond length in Bohr: {}\n".format(BOND_LENGTH_MIN))
file.write("Maximum bond length in Bohr: {}\n".format(BOND_LENGTH_MAX))
file.close()
        

create_folders_all_structures(path, all_structures)


logging.info("\n========================================\nEND OF PROGRAM\n========================================")
