import os
import sys
import logging

import numpy as np
import matplotlib.pyplot as plt 

import ase.io
from ase import Atoms, Atom, units
from ase.calculators.vasp import Vasp
from ase.calculators.emt import EMT
from ase.build import fcc110
from ase.build import fcc111
from ase.build import fcc100
from ase.build import surface
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import VelocityVerlet
from ase.constraints import FixAtoms
from ase.calculators.vasp.vasp import VaspChargeDensity

import amp
from amp import Amp
from amp import utilities
from amp.model import LossFunction
from amp.analysis import read_trainlog
from amp.analysis import plot_convergence
from amp.analysis import plot_sensitivity
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork


# ---------------------------------------------------------
# GLOBAL SETTINGS AND VARIABLES
# ---------------------------------------------------------

FILENAME = 'Al6.traj' #'al_large_longMD.traj'

logging.basicConfig(level=logging.INFO)

PATH_PLOT = "/home/nuss/01/bt702501/Dokumente/PlotsII_reloaded/"

T_GLOBAL = 300
DT_STEP_GLOBAL = 1

# ---------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------

def generate_data(count, filename='data.traj'):
    """Generates test or training data with a simple MD simulation."""
    if not os.path.exists(filename):
        traj = ase.io.Trajectory(filename, 'w')
        #atoms = fcc110('Pt', (2, 2, 2), vacuum=7.)
        #atoms.extend(Atoms([Atom('Cu', atoms[7].position + (0., 0., 2.5)),
        #             Atom('Cu', atoms[7].position + (0., 0., 5.))]))
        #atoms.set_constraint(FixAtoms(indices=[0, 2]))

        T = T_GLOBAL
        dt_step = DT_STEP_GLOBAL
        
        if filename == 'gold.traj' or filename == 'gold2.traj':
            atoms = fcc111('Au', (2,2,2), vacuum=7.0)
            atoms.set_chemical_symbols(['Au', 'Au', 'Au', 'Au', 'Pd', 'Pd', 'Pd', 'Pd'])
            T = 6000
            dt_step = 5
            print(atoms)
        elif filename == 'al.traj' or filename == 'al2.traj':
            atoms = fcc100('Al', (3,3,2), vacuum=10.0)
            T = 300
            dt_step = 2
        elif filename == 'al_large.traj' or filename == 'al_large_longMD.traj' or filename == 'Al6.traj':
            atoms = fcc100('Al', (4,6,6), vacuum=20.)
            T = 300
            dt_step = 2
        elif filename == 'Al6.traj':
            atoms = fcc100('Al', (4,6,6), vacuum=20.)
            T = 300
            dt_step = 2
        elif filename == 'Al8.traj':
            atoms = fcc100('Al', (4,6,8), vacuum=20.)
            T = 300
            dt_step = 2
        elif filename == 'al_large_12layers.traj':
            atoms = fcc100('Al', (4,6,12), vacuum=20.)
            T = 300
            dt_step = 2
        elif filename == 'MA.traj':
            with open("POSCAR_MA") as f:
                all_lines = f.readlines()
                all_lines = [x[:-1] for x in all_lines]
            all_atoms = all_lines[5].split()
            n_atoms = all_lines[6].split()
            atoms = []
            for atom, n in zip(all_atoms, n_atoms):
                atoms.append((atom, n))
            atom_types = ''
            for atom_type, n in atoms:
                atom_types += str(atom_type)
                if n != '1':
                    atom_types += str(n)
            lattice_vectors = np.array([np.float64(line.split()) for line in all_lines[2:5]])
            all_coordinates = np.array([np.float64(line.split()) for line in all_lines[8:]])

            atoms = Atoms(atom_types, pbc=True, positions=all_coordinates, cell=lattice_vectors)
            T = 300
            dt_step = 2 
         
        print(atoms)
        print(atoms.get_positions())
        print(atoms.get_chemical_symbols())

        atoms.set_calculator(Vasp(setups='recommended', npar=16, nsim=4))
        MaxwellBoltzmannDistribution(atoms, T * units.kB)
        dyn = VelocityVerlet(atoms, dt=dt_step * units.fs)
        dyn.run(50)

        energy = atoms.get_potential_energy()
        traj.write(atoms)
        densities = []
        energies = []
        energies.append(energy)
        # IMPORTANT: Density is already divided by volume!!!
        C = VaspChargeDensity(filename='CHGCAR')
        densities.append(C.chg)

        for step in range(count-1):
            logging.info("    Calculating step {}".format(step+1))
            dyn.run(5)
            traj.write(atoms)
            C = VaspChargeDensity(filename='CHGCAR')
            densities.append(C.chg)
            energies.append(atoms.get_potential_energy())
            print(energies[-1])
        traj.close()
        densities = np.array(densities)
        
    densities_numpy = 'densities_{}.npy'.format(filename[:-5])
    if not os.path.exists(densities_numpy):
        np.save(densities_numpy, densities, allow_pickle=True)
    else:
        densities = np.load(densities_numpy, allow_pickle=True)
    energies_numpy = 'energies_{}.npy'.format(filename[:-5])
    if not os.path.exists(energies_numpy):
        np.save(energies_numpy, energies, allow_pickle=True)
    else:
        energies = np.load(energies_numpy, allow_pickle=True)

    # Train-test-split
    train_images, test_images, train_densities, test_densities, train_energies, test_energies = train_test_split(
                                                    filename, densities, energies, fraction=0.8) 

    #densities_train_numpy = 'densities_train_{}.npy'.format(filename[:-5])
    #densities_test_numpy = 'densities_test_{}.npy'.format(filename[:-5])
    #if not os.path.exists(densities_train_numpy) or os.path.exists(densities_test_numpy):
    #    np.save(densities_train_numpy, train_densities)
    #    np.save(densities_test_numpy, test_densities)

    return (train_images, test_images, train_densities, test_densities, train_energies, test_energies)


def predict(train_images, test_images, calc):
    fig, ax = plt.subplots()

    actual_energies = []
    actual_densities = []
    predicted_energies = []    
    predicted_densities = []

    # TODO: Rework for loop --> Calculator might be reset at wrong times
    # Predicting on training data
    for i_dataset, atoms in enumerate(train_images):
        # Get actual energy
        actual_energy = atoms.get_potential_energy()
        actual_energies.append(actual_energy)
        # Get predicted energy
        atoms.set_calculator(calc)
        predicted_energy = atoms.get_potential_energy()
        predicted_energies.append(predicted_energy)
        
        ax.plot(actual_energy, predicted_energy, 'b.')
    
    # Predicting on test data
    for i_dataset, atoms in enumerate(test_images):
        # Get actual energy
        actual_energy = atoms.get_potential_energy()
        actual_energies.append(actual_energy)
        # Get predicted energy
        atoms.set_calculator(calc)
        predicted_energy = atoms.get_potential_energy()
        predicted_energies.append(predicted_energy)

        ax.plot(actual_energy, predicted_energy, 'r.')
    
    ax.set_xlabel('Actual energy / eV')
    ax.set_ylabel('Predicted energy / eV')
    fig.savefig(os.path.join(PATH_PLOT, 'parity000.png'))

    return (actual_energies, predicted_energies)
        
        

def train_test_split(images, densities, energies, fraction=0.8):
    """Randomly assigns 'fraction' of the images to a training set and
    (1-'fraction') to a test set. Returns two lists of ASE images
    and two lists of the respective densities
    
    Parameters
    ----------
    images: str
        Path to ASE trajectory (.traj)
    densities: numpy array
        Numpy array containing all densities
    fraction: float
        Portion of train_images to all images

    Returns
    -------
    train_images, test_images: list
        List of train and test images
    train_densities, test_densities: list
        Numpy array of train and test densities
    """
    images = ase.io.Trajectory(images, 'r')
    
    trainingsize = int(fraction * len(images))
    testsize = len(images) - trainingsize
    testindices = []
    while len(testindices) < testsize:
        next = np.random.randint(len(images))
        if next not in testindices:
            testindices.append(next)
    testindices.sort()
    trainindices = [index for index in range(len(images)) if index
                    not in testindices]
    train_images = [images[index] for index in trainindices]
    test_images = [images[index] for index in testindices]
    
    print("N Densities:", len(densities))
    print("Train indices:", trainindices)
    print("Test indices:", testindices)

    train_densities = np.array([densities[index] for index in trainindices])
    test_densities = np.array([densities[index] for index in testindices])

    train_energies = np.array([energies[index] for index in trainindices])
    test_energies = np.array([energies[index] for index in testindices]) 

    images.close()
    
    return train_images, test_images, train_densities, test_densities, train_energies, test_energies



def observer(model, vector, loss):
    """Function used for verbosity during training
       ERROR in amp --> Function not correctly implemented in current version"""
    print(vector[0])

# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------

if __name__ == '__main__':
    from amp.analysis import plot_parity_and_error

    # Generate training and test data
    logging.info("Generating training and test data")
    train_images, test_images, \
    train_densities, test_densities,\
    train_energies, test_energies\
         = generate_data(50, FILENAME)
    logging.info("Generation of training and test data finished!")
    print("Train densities:", train_densities.shape)
    print("Test densities:", test_densities.shape)
    print("Train energies:", train_energies.shape)
    print("Test energies:", test_energies.shape)
 
    
    #sys.exit(0)  
    # Training model
    logging.info("Starting Training")
    
    cores = {} 
    with open('mpd.hosts') as f:
        all_lines = f.readlines()
        for core in all_lines:
            core = core[:-1]
            if core not in cores.keys():
                cores[core] = 1
            elif core in cores.keys():
                cores[core] += 1
    print(cores) 

    calc = Amp(descriptor=Gaussian(),
               model=NeuralNetwork(hiddenlayers=(10, 10, 10)),
               cores=32,
               #envcommand='export PYTHONPATH=/tp_leppert/amp_package/amp',
               label='calc_{}'.format(FILENAME[:-5]))
    convergence = {'energy_rmse': 0.0009,
                   'energy_maxresid': 0.0018,
                   'force_rmse': 0.1,
                   'force_maxresid': 0.7}
    calc.model.lossfunction = LossFunction(convergence=convergence, force_coefficient=0.3)
    calc.train(images=train_images)
    #calc = Amp.load('calc_al_large.amp')
    logging.info("Training finished!")

    #print(calc.descriptor.fingerprints)
    calc_vasp = Vasp(setups='recommended', npar=16, nsim=4)

    energies = []
    forces = [] 
    for image in test_images:
        image.set_calculator(calc_vasp)
        energy = image.get_potential_energy()
        force = image.get_forces()
        energies.append(energy)
        forces.append(force)
    energies = np.array(energies)
    forces = np.array(forces)
    print(energies)
    np.save('energies_vasp_{}'.format(FILENAME[:-5]), energies)
    np.save('forces_vasp_{}'.format(FILENAME[:-5]), forces)


    energies = []
    forces = []
    for image in test_images:
        image.set_calculator(calc)
        energy = image.get_potential_energy()
        force = image.get_forces()
        energies.append(energy)
        forces.append(force)
    energies = np.array(energies)
    forces = np.array(forces)
    print(energies)
    np.save('energies_amp_{}'.format(FILENAME[:-5]), energies)
    np.save('forces_amp_{}'.format(FILENAME[:-5]), forces)


    # Testing model
    plot_parity_and_error(calc=calc, 
                          images=test_images,
                          plotfile_parity=os.path.join(PATH_PLOT, 'parity_{}.pdf'.format(FILENAME[:-5])),
                          plotfile_error=os.path.join(PATH_PLOT, 'error_{}.pdf'.format(FILENAME[:-5])),
                          overwrite=True)
    #actual_energies, predicted_energies = predict(train_images, test_images, calc)
    logdata = read_trainlog('calc_{}-log.txt'.format(FILENAME[:-5]))
    plot_convergence(data=logdata, plotfile=os.path.join(PATH_PLOT, 'convergence_{}.pdf'.format(FILENAME[:-5])))
    plot_sensitivity(calc=calc, images=FILENAME, plotfile=os.path.join(PATH_PLOT, 'sensitivity_{}.pdf'.format(FILENAME[:-5])))
    
