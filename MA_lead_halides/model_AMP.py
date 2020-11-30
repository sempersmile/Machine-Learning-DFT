import os
import shutil
import sys
import logging

import random
import numpy as np
import matplotlib.pyplot as plt 

import ase.io
from ase.io.vasp import write_vasp
from ase import Atoms, Atom, units
from ase.calculators.vasp import Vasp
from ase.calculators.emt import EMT
from ase.calculators.vasp.vasp import VaspChargeDensity
from ase.build import fcc110
from ase.build import fcc111
from ase.build import fcc100
from ase.build import surface
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.npt import NPT
from ase.md import VelocityVerlet
from ase.constraints import FixAtoms
from ase.dft.kpoints import monkhorst_pack

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

FILENAME = 'MA_2x2x2_XDATCAR.traj'
XDATCAR = 'XDATCAR-2x2x2'
POSCARS = 'POSCARS_6_07/'

PATH_PLOT = "/home/nuss/01/bt702501/Dokumente/Plots/"

logging.basicConfig(level=logging.INFO)

SAVE_CONTCAR = True

T_GLOBAL = 300
DT_STEP_GLOBAL = 1

# ---------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------

def generate_data(count, filename='data.traj'):
    """Generates test or training data with a simple MD simulation."""
    if not os.path.exists(filename):
        traj = ase.io.Trajectory(filename, 'w')
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
        elif filename == 'al_large.traj' or filename == 'al_large_longMD.traj' or filename == 'test.traj' or filename == 'test2.traj':
            atoms = fcc100('Al', (4,6,6), vacuum=20.)
            T = 300
            dt_step = 2
        elif filename == 'al_large_8layers.traj':
            atoms = fcc100('Al', (4,6,8), vacuum=20.)
            T = 300
            dt_step = 2
        elif filename == 'al_large_12layers.traj':
            atoms = fcc100('Al', (4,6,12), vacuum=20.)
            T = 300
            dt_step = 2
        elif filename == 'MA.traj' or filename == 'MA_2x2x2.traj':
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
            dt_step = 1
         
        atoms.set_calculator(Vasp(setups='gw',
                                  istart=0,
                                  encut=300,
                                  ismear=0,
                                  sigma=0.1,
                                  ediff=1.0E-05,
                                  ivdw=21,
                                  kpts=monkhorst_pack((2,2,2)),
                                  npar=16,
                                  nsim=2))
        #MaxwellBoltzmannDistribution(atoms, T * units.kB)
        #dyn = VelocityVerlet(atoms, dt=dt_step * units.fs)
        dyn = NPT(atoms, 1*units.fs, 350*units.kB, 0.001, 25*units.fs, 0.000001)
        dyn.run(500)

        if not os.path.exists("CONTCARS"):
            os.makedirs("CONTCARS")
        shutil.copyfile("CONTCAR", "CONTCARS/CONTCAR_0") 

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
            dyn.run(100)
            shutil.copyfile("CONTCAR", "CONTCARS/CONTCAR_{}".format(step+1)) 
            traj.write(atoms)
            C = VaspChargeDensity(filename='CHGCAR')
            densities.append(C.chg)
            energies.append(atoms.get_potential_energy())
            print(energies[-1])
        traj.close()
        densities = np.array(densities)
        
    # Train-test-split
    train_images, test_images= train_test_split(filename, fraction=0.8) 

    return (train_images, test_images)


def read_poscar(filename_poscar='MA_2x2x2_volume.traj'):
    if not os.path.exists(filename_poscar):
        traj = ase.io.Trajectory(filename_poscar, 'w') 
        for poscar in os.listdir(POSCARS):
            with open(os.path.join(POSCARS, poscar)) as f:
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
            dt_step = 1
         
            atoms.set_calculator(Vasp(setups='gw',
                                  istart=0,
                                  xc='PBE',
                                  encut=300,
                                  ismear=0,
                                  sigma=0.1,
                                  ediff=1.0E-06,
                                  ivdw=21,
                                  kpts=monkhorst_pack((2,2,2)),
                                  npar=16,
                                  nsim=4))
            atoms.get_potential_energy()
            traj.write(atoms)     
        traj.close()


def read_vasp_xdatcar(filename=XDATCAR, index=0):
    """Import XDATCAR file

       Reads all positions from the XDATCAR and returns a list of
       Atoms objects.  Useful for viewing optimizations runs
       from VASP5.x

       Constraints ARE NOT stored in the XDATCAR, and as such, Atoms
       objects retrieved from the XDATCAR will not have constraints set.
    """
    f = open(filename, 'r')
    images = list()

    cell = np.eye(3)
    atomic_formula = str()

    while True:
        comment_line = f.readline()
        if "Direct configuration=" not in comment_line:
            try:
                lattice_constant = float(f.readline())
            except Exception:
                # XXX: When would this happen?
                break

            xx = [float(x) for x in f.readline().split()]
            yy = [float(y) for y in f.readline().split()]
            zz = [float(z) for z in f.readline().split()]
            cell = np.array([xx, yy, zz]) * lattice_constant

            symbols = f.readline().split()
            numbers = [int(n) for n in f.readline().split()]
            total = sum(numbers)

            atomic_formula = ''.join('{:s}{:d}'.format(sym, numbers[n])
                                     for n, sym in enumerate(symbols))

            f.readline()

        coords = [np.array(f.readline().split(), np.float)
                  for ii in range(total)]

        image = Atoms(atomic_formula, cell=cell, pbc=True)
        image.set_scaled_positions(np.array(coords))
        images.append(image)
    f.close()

    if not index:
        return images
    else:
        return images[index]


def read_xdatcar(xdatcar=XDATCAR, filename=FILENAME, k=10):
    all_atoms_ = read_vasp_xdatcar(xdatcar, index=0) 
    
    if not os.path.exists(filename):
        traj = ase.io.Trajectory(filename, 'w')
        for atoms_ in random.choices(all_atoms_[2000:], k=k):
            print(atoms_)
            positions = atoms_.get_positions()

            # READ ATOMS PARAMS FROM POSCAR_MA FILE
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

            atoms = Atoms(atom_types, pbc=True, positions=positions, cell=lattice_vectors)

            #atoms.set_calculator(Vasp(setups='gw', npar=8, nsim=2))
            atoms.set_calculator(Vasp(setups='gw', # checked
                                      xc='PBE',
                                      istart=0, # checked
                                      encut=300, # checked
                                      ismear=0, # checked
                                      sigma=0.1, 
                                      ediff=1.0E-07,
                                      ivdw=21,
                                      kpts=monkhorst_pack((2,2,2)), # checked
                                      npar=16, # checked
                                      nsim=4)) # checked))
            #write_vasp('CONTCAR', atoms)     
            atoms.get_potential_energy()
            #atoms.get_forces()
            traj.write(atoms)     
        traj.close()
    train_images, test_images = train_test_split(filename, fraction=0.5)# fraction originally: 0.8 

    return (train_images, test_images)


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
        
        

def train_test_split(images, fraction=0.8):
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
    
    images.close()
     
    return train_images, test_images


# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------

if __name__ == '__main__':
    from amp.analysis import plot_parity_and_error
    """
    # Generate training and test data
    logging.info("Generating training and test data")
    train_images, test_images = read_xdatcar(XDATCAR, FILENAME, 100)
    #train_images, test_images = generate_data(20, FILENAME) 
    logging.info("Generation of training and test data finished!")
 
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
               label='calc_{}'.format(FILENAME[:-5]))
    convergence = {'energy_rmse': 0.0018,
                   'energy_maxresid': 0.004,
                   'force_rmse': 0.07,
                   'force_maxresid': 0.3}
    calc.model.lossfunction = LossFunction(convergence=convergence, force_coefficient=0.3)
    calc.train(images=train_images)
    """
    #read_poscar(FILENAME)
    train_images, test_images = read_xdatcar(XDATCAR, k=20)

    #= ase.io.Trajectory(FILENAME, 'r')
    calc = Amp.load('calc_MA_2x2x2.amp')
    
    logging.info("Training finished!")
    


    calc_vasp = Vasp(setups='gw',
                              xc='PBE',
                              istart=0, 
                              encut=300,
                              ismear=0, 
                              sigma=0.1, 
                              ediff=1.0E-07,
                              ivdw=21,
                              kpts=monkhorst_pack((2,2,2)),
                              npar=16,
                              nsim=4)
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
    #logdata = read_trainlog('calc_{}-log.txt'.format(FILENAME[:-5]))
    #plot_convergence(data=logdata, plotfile=os.path.join(PATH_PLOT, 'convergence_{}.pdf'.format(FILENAME[:-5])))
    #plot_sensitivity(calc=calc, images=FILENAME, plotfile=os.path.join(PATH_PLOT, 'sensitivity_{}.pdf'.format(FILENAME[:-5])))
    
