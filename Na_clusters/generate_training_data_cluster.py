import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from math import pi

logging.basicConfig(level=logging.INFO)

#######################################
# Setting global parameters
#######################################

RANDOM = True

N_STRUCTURES = int(input("Number of structures to generate? "))# Total number of Na-structures
N_ATOMS = 20 # Number of Na-atoms per Na-structure
PATH = "/tp_leppert/bt702501/Calculated_Data/Parameter_Set_79/"

if N_ATOMS == 4:
    BOND_LENGTH_MIN_ANGSTROM = 5.77 * 0.529177
    BOND_LENGTH_MAX_ANGSTROM = 6.79 * 0.529177
elif N_ATOMS == 6:
    BOND_LENGTH_MIN_ANGSTROM = 6.50 * 0.529177
    BOND_LENGTH_MAX_ANGSTROM = 7.03 * 0.529177
elif N_ATOMS == 20:
    BOND_LENGTH_MIN_ANGSTROM = 5.3 * 0.529177
    BOND_LENGTH_MAX_ANGSTROM = 5.3 * 0.529177



BOND_LENGTH_MIN = BOND_LENGTH_MIN_ANGSTROM * 1.889725989
BOND_LENGTH_MAX = BOND_LENGTH_MAX_ANGSTROM * 1.889725989
BOX_SIZE = 9.7 # TODO: Orig: 9.0# Box size in Bohr (--> Box in range -BOX_SIZE to +BOX_SIZE)
DIMENSION = 3 # Specify dimension of problem

UNCERTAINTY = 0.1 * BOND_LENGTH_MIN

#######################################
# Functions
#######################################


def translate(coords):
    trans = np.random.uniform(-2, 2, (3))
    for i, atom in enumerate(coords):
        coords[i,:] = coords[i,:] + trans 
    return coords


def rotate_x(coords):
    alpha = np.random.uniform(0, 2*pi, (1))
    for i, atom in enumerate(coords):
        coords[i,:] = np.dot(np.array([[1, 0, 0],
                                       [0, np.cos(alpha), -np.sin(alpha)],
                                       [0, np.sin(alpha), np.cos(alpha)]]),
                             coords[i, :])
    return coords


def rotate_y(coords):
    alpha = np.random.uniform(0, 2*pi, (1))
    for i, atom in enumerate(coords):
        coords[i,:] = np.dot(np.array([[np.cos(alpha), 0, np.sin(alpha)],
                                       [0, 1, 0],
                                       [-np.sin(alpha), 0, np.cos(alpha)]]),
                             coords[i, :])
    return coords


def rotate_z(coords):
    alpha = np.random.uniform(0, 2*pi, (1))
    for i, atom in enumerate(coords):
        coords[i,:] = np.dot(np.array([[np.cos(alpha), -np.sin(alpha), 0],
                                       [np.sin(alpha), np.cos(alpha), 0],
                                       [0, 0, 1]]),
                             coords[i, :])
    return coords


def get_random_structure(n_atoms=N_ATOMS, bond_length_min=BOND_LENGTH_MIN,
                       bond_length_max=BOND_LENGTH_MAX,dim=DIMENSION,box_size=BOX_SIZE): 
    coords = np.zeros((n_atoms,dim))
    in_box = False
    while in_box == False:
        in_box = True

        if n_atoms == 3: 
            # Triangular shape
            xyz = 1/(np.sqrt(2)) * np.random.uniform(bond_length_min, bond_length_max, (n_atoms))
            coords[0,:] = np.array([xyz[0], 0, 0])
            coords[1,:] = np.array([0, xyz[1], 0])
            coords[2,:] = np.array([0, 0, xyz[2]])
            
            coords = translate(coords)
            coords = rotate_x(coords)
            coords = rotate_y(coords)
            coords = rotate_z(coords) 

            for counter, atoms in enumerate(coords[:,:], 0):
                assert np.linalg.norm(coords[counter,:]-coords[counter-1,:]) >= bond_length_min
                assert np.linalg.norm(coords[counter,:]-coords[counter-1,:]) <= bond_length_max
                if np.max(np.absolute(coords[counter,:])) > BOX_SIZE:
                    in_box = False

        elif n_atoms == 4:
            # Trapezoidal shape
            coords[0,:] = np.array([0, 0, 0]) 
            coords[1,:] = np.array([bond_length_min, 0, 0])
            coords[2,:] = np.array([bond_length_min/2, bond_length_max*np.sin(np.arccos(bond_length_min/(2*bond_length_max))), 0])
            coords[3,:] = np.array([bond_length_min/2, -bond_length_max*np.sin(np.arccos(bond_length_min/(2*bond_length_max))), 0])
            
            coords += np.random.uniform(-UNCERTAINTY, UNCERTAINTY, coords.shape)

            #coords = translate(coords)
            #coords = rotate_x(coords)
            #coords = rotate_y(coords)
            #coords = rotate_z(coords) 

            for counter, atoms in enumerate(coords[:,:], 0):
                # No complete assertion for this case!
                #assert np.linalg.norm(coords[counter,:]-coords[counter-1,:]) >= bond_length_min - 0.000001
                #assert np.linalg.norm(coords[counter,:]-coords[counter-1,:]) <= bond_length_max + 0.000001
                if np.max(np.absolute(coords[counter,:])) > BOX_SIZE:
                    in_box = False

        elif n_atoms == 6:
            r = bond_length_max / (2*np.sin(2*pi/10))
            z = np.sqrt(bond_length_min**2 - r**2)

            coords[0,:] = np.array([0, 0, 0]) 
            coords[1,:] = np.array([r*np.sin(2*pi*1/5), r*np.cos(2*pi*1/5), -2.5475])
            coords[2,:] = np.array([r*np.sin(2*pi*2/5), r*np.cos(2*pi*2/5), -2.5475])
            coords[3,:] = np.array([r*np.sin(2*pi*3/5), r*np.cos(2*pi*3/5), -2.5475])
            coords[4,:] = np.array([r*np.sin(2*pi*4/5), r*np.cos(2*pi*4/5), -2.5475])
            coords[5,:] = np.array([r*np.sin(2*pi*5/5), r*np.cos(2*pi*5/5), -2.5475])
            
            coords += np.random.uniform(-UNCERTAINTY, UNCERTAINTY, coords.shape)

            #coords = translate(coords)
            #coords = rotate_x(coords)
            #coords = rotate_y(coords)
            #coords = rotate_z(coords) 

            for counter, atoms in enumerate(coords[:,:], 0):
                # No complete assertion for this case!
                #assert np.linalg.norm(coords[counter,:]-coords[counter-1,:]) >= bond_length_min - 0.000001
                #assert np.linalg.norm(coords[counter,:]-coords[counter-1,:]) <= bond_length_max + 0.000001
                if np.max(np.absolute(coords[counter,:])) > BOX_SIZE:
                    in_box = False

        elif n_atoms == 10:
            # Assuming an equilateral pyramid
            h = np.sqrt(bond_length_min**2 - 1/3 * bond_length_min**2)
            r = np.sqrt(bond_length_min**2 - h**2)

            print(r)
            print(h)
            print(bond_length_min)

            z0 = 7
            z1 = -h

            x0 = 0
            x1 = r*np.sin(2*pi*1/3) 
            x2 = r*np.sin(2*pi*2/3) 
            x3 = r*np.sin(2*pi*3/3) 
            
            y0 = -3
            y1 = r*np.cos(2*pi*1/3) 
            y2 = r*np.cos(2*pi*2/3) 
            y3 = r*np.cos(2*pi*3/3) 

            v1 = np.array([x1, y1, z1])
            v2 = np.array([x2, y2, z1])
            v3 = np.array([x3, y3, z1])
            
            coords[0,:] = np.array([x0, y0, z0])

            coords[1,:] = coords[0,:] + v1 
            coords[2,:] = coords[0,:] + v2
            coords[3,:] = coords[0,:] + v3

            coords[4,:] = coords[0,:] + v1 + v1
            coords[5,:] = coords[0,:] + v1 + v2
            coords[6,:] = coords[0,:] + v1 + v3
            coords[7,:] = coords[0,:] + v2 + v2
            coords[8,:] = coords[0,:] + v2 + v3
            coords[9,:] = coords[0,:] + v3 + v3 


            coords += np.random.uniform(-UNCERTAINTY, UNCERTAINTY, coords.shape)
            
            for i, elem in enumerate(coords):
                print("Distance from", i, "to", i-1, ":", np.linalg.norm(coords[i] - coords[i-1]))

            #coords = translate(coords)
            #coords = rotate_x(coords)
            #coords = rotate_y(coords)
            #coords = rotate_z(coords) 

            for counter, atoms in enumerate(coords[:,:], 0):
                if np.max(np.absolute(coords[counter,:])) > BOX_SIZE:
                    in_box = False



        elif n_atoms == 20:
            # Assuming an equilateral pyramid
            h = np.sqrt(bond_length_min**2 - 1/3 * bond_length_min**2)
            r = np.sqrt(bond_length_min**2 - h**2)

            print(r)
            print(h)
            print(bond_length_min)

            z0 = 7.5
            z1 = -h

            x0 = 0
            x1 = r*np.sin(2*pi*1/6) 
            x2 = r*np.sin(2*pi*3/6) 
            x3 = r*np.sin(2*pi*5/6) 
            
            y0 = 0
            y1 = r*np.cos(2*pi*1/6) 
            y2 = r*np.cos(2*pi*3/6) 
            y3 = r*np.cos(2*pi*5/6) 

            v1 = np.array([x1, y1, z1])
            v2 = np.array([x2, y2, z1])
            v3 = np.array([x3, y3, z1])
            
            coords[0,:] = np.array([x0, y0, z0])

            coords[1,:] = coords[0,:] + v1 
            coords[2,:] = coords[0,:] + v2
            coords[3,:] = coords[0,:] + v3

            coords[4,:] = coords[0,:] + v1 + v1
            coords[5,:] = coords[0,:] + v1 + v2
            coords[6,:] = coords[0,:] + v1 + v3
            coords[7,:] = coords[0,:] + v2 + v2
            coords[8,:] = coords[0,:] + v2 + v3
            coords[9,:] = coords[0,:] + v3 + v3 

            coords[10,:] = coords[0,:] + v1 + v1 + v1
            coords[11,:] = coords[0,:] + v1 + v1 + v2
            coords[12,:] = coords[0,:] + v1 + v1 + v3
            coords[13,:] = coords[0,:] + v1 + v2 + v2
            coords[14,:] = coords[0,:] + v1 + v2 + v3
            coords[15,:] = coords[0,:] + v1 + v3 + v3
            coords[16,:] = coords[0,:] + v2 + v2 + v2
            coords[17,:] = coords[0,:] + v2 + v2 + v3
            coords[18,:] = coords[0,:] + v2 + v3 + v3
            coords[19,:] = coords[0,:] + v3 + v3 + v3
            


            coords += np.random.uniform(-UNCERTAINTY, UNCERTAINTY, coords.shape)
            
            for i, elem in enumerate(coords):
                print("Distance from", i, "to", i-1, ":", np.linalg.norm(coords[i] - coords[i-1]))

            #coords = translate(coords)
            #coords = rotate_x(coords)
            #coords = rotate_y(coords)
            #coords = rotate_z(coords) 

            for counter, atoms in enumerate(coords[:,:], 0):
                if np.max(np.absolute(coords[counter,:])) > BOX_SIZE:
                    in_box = False


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
