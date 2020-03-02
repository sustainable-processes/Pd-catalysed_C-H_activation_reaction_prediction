#! /usr/bin/python
import os
import math
import copy
import numpy as np


atom_string = ['H', 'B', 'C', 'N', 'O', 'F',  # list of atoms corresponding
        'Si', 'P', 'S', 'Cl', 'Br', 'I']

atom_list = {1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F',  # list of atoms corresponding
             14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 3: 'Li', 35: 'Br', 53: 'I', 46: 'Pd'}

"""
The dictionary defining covalent atomic radii for elements

"""

Atom_radii = {'N': 1.89, 'H': 1.2, 'He': 0.31, 'Li': 1.82, 'Be': 1.12, 'Ne': 0.38, 'Na': 1.9, 'Mg': 1.45, 'Al': 1.18,
              'Si': 1.11, 'Ar': 0.71, 'K': 2.43, 'Ca': 1.94, 'Sc': 1.84,
              'Ti': 1.76, 'V': 1.71, 'Cr': 1.66, 'Mn': 1.61, 'Co': 1.52, 'Ni': 1.49, 'Cu': 1.45, 'Zn': 1.42, 'Ga': 1.36,
              'Ge': 1.25, 'As': 1.14, 'Se': 1.03, 'Kr': 0.88, 'Rb': 2.65, 'Sr': 2.19,
              'Y': 2.12, 'Zr': 2.06, 'Nb': 1.98, 'Mo': 1.9, 'Tc': 1.85, 'Rh': 1.73, 'Pd': 1.69, 'Ag': 1.65, 'Cd': 1.61,
              'In': 1.56, 'Sn': 1.45, 'Sb': 1.33, 'Te': 1.23, 'Xe': 1.08,
              'Cs': 2.98, 'Ba': 2.53, 'La': 2.51, 'Ce': 2.49, 'C': 1.85, 'O': 2.294, 'I': 1.15, 'Cl': 2.38, 'Ru': 1.78,
              'Br': 0.94, 'F': 1.73, 'P': 0.98, 'S': 0.88,
              'B': 0.87, 'Fe': 1.56}


######## Data taken from: International tables for Crystallography (2006), Vol. C, Chapter 9.5, pp. 790 - 811 ##############
"""
The dictionary containing interatomic distances thresholds used to define bonds and their multiplicities for all Element-Element pairs

Format: Element1Element2: [maximum single bond length, maximum aromatic (conjugated double) bond length,
                            maximum double bond length, maximum triple bond length]
                            
If the length of the value is one, only single bond threshold is given                   
                            
"""

Connectivity = {'CC': [1.60, 1.43, 1.35, 1.20], 'BB': [1.77], 'BBr': [2.1], 'BC': [1.62], 'BCl': [1.85], 'BI': [2.25], 'BF': [1.38], 'BN': [1.61], 'BO': [1.48], 'BP': [1.93],
                'BS': [1.93], 'BrBr': [2.54], 'BrC': [1.97], 'BrI': [2.70], 'BrN': [1.85], 'BrO': [1.59], 'BrP': [2.37], 'BrS': [2.45], 'BrSe': [2.62], 'BrSi': [2.28], 'BrTe': [3.1],
                'CCl': [1.74], 'CF': [1.43], 'CH': [1.15], 'CI': [2.18], 'CN': [1.55, 1.361, 1.315, 1.15], 'CO': [1.50, 1.30, 1.28, 'N/A'], 'CP': [1.86], 'CS': [1.86, 1.724, 1.69, 'N/A'],
                'CSe': [1.97], 'CSi': [1.89], 'CTe': [2.16], 'ClCl': [2.31], 'ClI': [2.58], 'ClN': [1.76], 'ClO': [1.42], 'ClP': [2.05], 'ClS': [2.1], 'ClSe': [2.25], 'ClSi': [2.1], 'ClTe': [2.52],
                'FN': [1.41], 'FP': [1.58], 'FS': [1.64], 'FSi': [1.70], 'FTe': [2.0], 'HN': [1.1], 'HO': [1.1], 'II': [2.93], 'HS': [1.45], 'IN': [2.25], 'IO': [2.17], 'IP': [2.50], 'IS': [2.75],
                'ITe': [2.95], 'NN': [1.45, 1.32, 1.24, 1.10], 'NO': [1.46, 'N/A', 1.24, 'N/A'], 'NP': [1.73, 'N/A', 1.60, 'N/A'], 'NS': [1.71, 'N/A', 1.56, 'N/A'], 'NSe': [1.86, 'N/A', 1.795, 'N/A'],
                'NSi': [1.76], 'NTe': [2.25], 'OO': [1.50], 'OP': [1.70, 'N/A', 1.51, 'N/A'], 'OS': [1.60, 'N/A', 1.44, 'N/A'], 'OSe': [2.0, 'N/A', 1.60, 'N/A'], 'OSi': [1.70],
                'OTe': [2.14], 'PP': [2.26, 'N/A', 2.05, 'N/A'], 'PS': [2.30, 'N/A', 1.95, 'N/A'], 'PSe': [2.45, 'N/A', 2.10, 'N/A'], 'PSi': [2.26], 'PTe': [2.50, 'N/A', 2.35, 'N/A'], 'SS': [2.10],
                'SSe': [2.20], 'SSi': [2.15], 'STe': [2.45], 'SeSe': [2.35], 'SeTe': [2.58], 'SiSi': [2.40], 'TeTe': [2.75]}

Coordinative_bonds = {'CI': 2.40}

electronegativity = {'F': 4.0, 'O': 3.5, 'N': 3.0, 'Cl': 3.2, 'S': 2.6, 'Br': 3.0, 'I': 2.7}

heavy_atoms = ['I', 'Br', 'Cl', 'Cu', 'Pd']

gaussian_pseudo = {'I':'MWB46', 'Cu': 'MDF10', 'Pd': 'MWB28'}

local_settings = ['6', 'export LD_LIBRARY_PATH=/opt/OpenBLAS/lib:/opt/mpich-3.1.3/lib:/opt/mopac',
                  '/opt/mpich-3.1.3/bin/mpiexec -np 6 /home/mk787/nwchem/bin/nwchem']
server_settings = ['6', 'export LD_LIBRARY_PATH=/opt/mpich-3.1.3/lib',
                   '/opt/mpich-3.1.3/bin/mpiexec -np 6 /home/mk787/nwchem/bin/nwchem']
mopac_settings = ['12', 'export LD_LIBRARY_PATH=/opt/mopac', '/opt/mopac/MOPAC2012.exe']


class Atom:

    """
Atom Class defining Atom class instances.

The required arguments are:
            label (atomic element)
            position (list of X, Y and Z coordinates; len(position) = 3

Methods available:
            distance(self, atom2) - defines the distance between self and atom2
            check_connected(self, atom2) - checks whether self is connected with atom2 using atom.radius values

    """

    def __init__(self, label, position):
        self.label = label
        self.position = [float(position[0]), float(position[1]), float(position[2])]
        self.number = 0
        self.type = "N/A"
        self.aromaticity = False
        self.Hattached = []
        self.connectivity_number = "N/A"
        self.radius = Atom_radii[label]

    def distance(self, atom2):
        return math.sqrt((atom2.position[0] - self.position[0]) ** 2 + (atom2.position[1] - self.position[1]) ** 2 + (
        atom2.position[2] - self.position[2]) ** 2)

    def check_connected(self, atom2):                           # Checks if Atom2 is connected to the atom1
        if self.distance(atom2) < (self.radius + atom2.radius) * 0.55:
            return True
        else:
            return False

class Bond:

    """

Bond class defining a bond between two atoms

The required arguments are:
        atom1 - first atom instance
        atom2 - second atom instance
        order - bond order

    """

    def __init__(self, atom1, atom2, order):
        self.order = order
        self.atom1 = atom1
        self.atom2 = atom2
        self.length = atom1.distance(atom2)

class Structure:

    """

The structure class defining a 3D system of atoms in space

The required arguments:
        atoms - a list of atom instances in the structure

The methods available:
        1) add_atoms(self, atoms)
        Adds a list of atoms (atoms) to the structure (self) updating their sequential atomic numbers

        2) remove_atoms(self, atoms)
        Removes a list of atoms (atoms) from the structure (self) updating the atomic numbers in the final structure

        3) center_at_atom(self, atom1)
        Translate the system (self) setting new origin to the coordinates of atom1

        4) centre_and_rotate(self, atom1, atom2)
        Translate and rotate the system so the atom1-atom2 vector is along 0X axis,
        atom1 is the new centre of coordinates

        5) conformers(self, flconf)
        Returns conformers of the self Structure parsing the MOL2 format file with multiple conformations

        6) Hconnected(self)
        Function that updates the self instance by updating hydrogen atoms connected to all heavy atoms

        7) connected_to_atom(self, atom)
        Returns a list of Atom objects in self connected to atom.

        8) nwchem_basis_input(self, input_open, light_atoms_basis, heavy_atoms_list, light_atoms_list)
        Auxilary function that writes basis set section of an NWChem input (input_open),
        assigning DFT basis set (light_atoms_basis)
        and ECP basis set to light atoms (light_atoms_list) and heavy atoms (heavy_atoms_list), respectively

        9) export_nwchem(self, charge, scratch, cluster)
        Main function to generate NWChem inputs for self structure, using charge of the structure,
        scratch directory name and cluster options as parameters

        10) export_nwchem_modular(self, charge, scratch, cluster, solvent, task)
        More general function allowing NWChem input creation using different solvent and task as additional parameters

        11) export_nwchem_fukui(self, charge)
        Function to generate NWChem input aimed at Fukui index calculations using default DFT options
        (structure optimisaiton (B3LYP/6-31g(d,p)) followed by frequency analysis, solvent = THF)

        12) export_nwchem_initial(self, scratch, cluster)
        Quick NWChem optimisation (max number of steps is limited to 5)

        13) export_nwchem_initial_charge(self, charge, scratch, cluster)
        Quick NWChem optimisation (max number of steps is limited to 5); applicable to charged spieces

        14) frequency_redo_submit(self, charge, scratch, cluster)
        Generate and submit NWChem calculation for the structure previously failed to converge to a local minimum

        15) load_checker(self)
        Auxilary function to check whether the NWChem output exists in the current directory

        16) reaction_centre(self)
        Auxilary function that updates the self instance with reaction centre determined from the filename

        17) connectivity_VDW(self)
        Function that updates the self Structure with the connectivity information using VDW radii returning all
        bonds as single only

        18) connectivity_by_distance(self)
        Function that updates the self Structure with the connectivity using Connectivity dictionary to determine bond
        multiplicities

        19) radius_SMD_read(self)
        Function that parses NWChem output extracting SMD radii and updates all atoms of the Structure

        20) connectivity_sdf(self)
        Function that updates the structure with the connectivity from SDF file bonding information

        21) connectivity_mol2(self)
        Function that updates the structure with the connectivity from MOL2 file bonding information

        22) initial_rename(self)
        Auxilary method that updates the structure name adding 'initial' to it

        23) determine_atom_by_identity(self,identity)
        Function that finds the atom object using its identity (element + number) returning the atom instance

        24) determine_atom_by_number(self,number)
        Function that finds the atom object using its sequential number returning the atom instance

        25) Pd_ipso_complex(self, atom_to_add, atom1, atom2, atom_added, H_added, substituent_H, distance)
        Function that builds possible Pd intermediates as ipso complexes from the Structure specifying
        - atom in the structure where Pd is added (atom_to_add)
        - atom1 and atom2 - two atoms linked to atom_to_add in the structure
        - Pd atom instance (atom_added)
        - atom in the Pd species added linked to Pd (H_added)
        - whole Pd species added as a structure instance (Substituent_H)
        - distance between atom in the structure and Pd atom

        26) add_substituent(self, substituent_H, atom_to_add, H_to_add, atom_added, H_added, atom_1, atom_2, distance)
        General function to add substituents to the self structure taking parameters:
        - added structure (Substituent_H)
        - atom in the original structure where the addition happenes (atom_to_add)
        - Hydrogen atom in the original structure that gets moved (H_to_add)
        - First added atom in the added structure that forms a bond with the original structure (atom_added)
        - Second atom, linked to the first added atom, in the added startucture (H_added)
        - atom1, atom2 - two atoms linked to the atom_to_add, apart from H
        - distance between the original structure and added structure


    """

    def __init__(self, atoms):
        self.atoms = atoms
        self.charge = "N/A"
        self.name = "N/A"
        self.energy = "N/A"
        self.connectivity = []
        self.solvent = "thf"
        self.Ecorr = None
        self.retry = "N/A"

    def add_atoms(self,atoms):
        self.atoms.sort(key = lambda x: x.number, reverse=False)
        atom_list = self.atoms
        atom_number = len(atom_list) + 1
        for i in atoms:
            i.number = atom_number
            atom_list.append(i)
            atom_number += 1
        new_structure = Structure(atom_list)
        return new_structure

    def remove_atoms(self,atoms):
        self.atoms.sort(key = lambda x: x.number, reverse=False)
        atom_list = self.atoms
        atom_list_remove = []
        for i in atoms:
            for j in atom_list:
                if i == j:
                    atom_list_remove.append(j)
        atom_list_new = []
        for i in atom_list:
            if not i in atom_list_remove:
                atom_list_new.append(i)
        atom_list_new.sort(key = lambda x: x.number, reverse=False)
        atom_number = 1
        for ii in atom_list_new:
            ii.number = atom_number
            atom_number += 1
        structure_new = Structure(atom_list_new)
        structure_new.name = self.name
        return structure_new

    def center_at_atom(self, atom1):
        translation = [atom1.position[0], atom1.position[1], atom1.position[2]]
        for i in self.atoms:
            i.position = vector_subtract(i.position, translation)

    def center_and_rotate(self, atom1, atom2):
        translation = [atom1.position[0], atom1.position[1], atom1.position[2]]
        for i in self.atoms:
            i.position = vector_subtract(i.position, translation)
        vector_new = [1, 0, 0]
        bond_old = vector_bond(atom2, atom1)
        theta = angle(vector_elementary(bond_old), vector_new)
        axis = vector_mult(vector_elementary(bond_old), vector_new)

        for j in self.atoms:
            j.position = vector_rotated(j.position, axis, theta)

    def conformers(self, flconf):  # returning conformers of a structure from the mol2 file
        confopen = open(flconf, "r")
        Conformer_list = []
        size = len(self.atoms)
        j = 0
        confnumb = 1
        check = 0
        for line in confopen.readlines():
            if "@<TRIPOS>ATOM" in line:
                check = 1
                j = 0
            elif check == 1:
                if j == 0:
                    line = line.split()
                    conformer = copy.deepcopy(self)
                    conformer.name = self.name + "_conf" + str(confnumb)
                    conformer.atoms[0].position = [float(line[2]), float(line[3]), float(line[4])]
                    conformer.charge = 0
                    j += 1
                elif j == size - 1:
                    line = line.split()
                    conformer.atoms[j].position = [float(line[2]), float(line[3]), float(line[4])]
                    Conformer_list.append(conformer)
                    confnumb += 1
                    j += 1
                    check = 0
                else:
                    line = line.split()
                    conformer.atoms[j].position = [float(line[2]), float(line[3]), float(line[4])]
                    j += 1
        return Conformer_list

    def Hconnected(self):  # auxillary function which finds protons tructures.[8]
        for j in self.atoms:
            if j.type == "heavy":
                Hlst = []
                for jj in self.atoms:
                    if jj.type == "light":
                        if jj.check_connected(j):
                            Hlst.append(jj)
                j.Hattached = Hlst

    def connected_to_atom(self, atom):
        connected = []
        for i in self.connectivity:
            if i.atom1 == atom:
                if i.atom2.type == 'heavy':
                    connected.append(i.atom2)
            elif i.atom2 == atom:
                if i.atom1.type == 'heavy':
                    connected.append(i.atom1)
        return connected

    def nwchem_basis_input(self, input_open, light_atoms_basis, heavy_atoms_list, light_atoms_list):
        if len(heavy_atoms_list) == 0:
            input_open.writelines("""basis spherical
* library """ + light_atoms_basis + """
end
""")
        else:
            input_open.write("basis spherical\n")
            for ii in light_atoms_list:
                input_open.write(ii + ' library ' + light_atoms_basis + '\n')
            for jj in heavy_atoms_list:
                input_open.write(jj + ' library crenbl_ecp\n')
            input_open.writelines("""end
ecp
""")
            for jj in heavy_atoms_list:
                input_open.write(jj + ' library crenbl_ecp\n')
            input_open.write('end\n')

    def export_nwchem(self, charge, scratch, cluster):                            # Normal NWChem structure optimisaiton (B3LYP/6-31g(d,p)) followed by frequency analysis, solvent = THF
        heavy_atoms_list = []
        light_atoms_list = []
        name_short = str(os.path.basename(self.name))
        nw_input = self.name + ".nw"
        input_open = open(nw_input, "w")
        input_open.write("start " + name_short + "\n")
        input_open.writelines(""" memory total 5000 mb
 scratch_dir """ + scratch + """
 title "optimisation_vacuum"

charge """ + str(charge) + """
geometry noautosym
""")
        for j in self.atoms:
            input_open.write(j.label + " " + str(j.position[0]) + " " + str(j.position[1]) + " " + str(j.position[2]) + "\n")
            if j.label in heavy_atoms:
                if not j.label in heavy_atoms_list:
                    heavy_atoms_list.append(j.label)
            else:
                if not j.label in light_atoms_list:
                    light_atoms_list.append(j.label)
        input_open.writelines("""end
 driver
  loose
  maxiter 30
 end
 dft
  xc b3lyp
  direct
  iterations 100
  grid xfine
  end
    """)
        self.nwchem_basis_input(input_open,'6-31g**',heavy_atoms_list, light_atoms_list)
        input_open.writelines("""
 task dft optimize

 title "frequencies"
 task dft frequencies

  title "solvation_energy"
  cosmo
  do_cosmo_smd true
  solvent thf
  end
  task dft
""")
        input_open.close()
        if cluster == "darwin":
            self.nwchem_submit_darwin()
        elif cluster == "local":
            self.nwchem_submit()
        elif cluster == 'darwin_liwei':
            self.nwchem_submit_liwei()

    def export_nwchem_modular(self, charge, scratch, cluster, solvent, task):
        heavy_atoms_list = []
        light_atoms_list = []
        name_short = str(os.path.basename(self.name))
        nw_input = self.name + ".nw"
        input_open = open(nw_input, "w")
        input_open.write("start " + name_short + "\n")
        input_open.writelines(""" memory total 5000 mb
 scratch_dir """ + scratch + """
 title "frequencies"

charge """ + str(charge) + """
geometry noautosym
""")
        for j in self.atoms:
            input_open.write(j.label + " " + str(j.position[0]) + " " + str(j.position[1]) + " " + str(j.position[2]) + "\n")
            if j.label in heavy_atoms:
                if not j.label in heavy_atoms_list:
                    heavy_atoms_list.append(j.label)
            else:
                if not j.label in light_atoms_list:
                    light_atoms_list.append(j.label)
        input_open.writelines("""end
 dft
  xc ssb-d
  direct
  iterations 100
  grid xfine
  end
  """)
        self.nwchem_basis_input(input_open,'cc-pvdz', heavy_atoms_list, light_atoms_list)
        input_open.writelines("""
 task dft freq

 title "optimisation"
 driver
  loose
  maxiter 50
 end
 task dft """ + task + """

 title "optimisation"
 task dft freq

 title "larger basis set"
 dft
 xc ssb-d
 direct
 iterations 100
 grid xfine
 end
 cosmo
 do_cosmo_smd true
  solvent """ + solvent + """
  end

  task dft
""")
        self.nwchem_basis_input(input_open,'cc-pvtz',heavy_atoms_list,light_atoms_list)
        input_open.close()
        if cluster == "darwin":
            self.nwchem_submit_darwin()
        elif cluster == "local":
            self.nwchem_submit()
        elif cluster == 'darwin_liwei':
            self.nwchem_submit_liwei()

    def export_nwchem_fukui(self, charge):  # Normal NWChem structure optimisaiton (B3LYP/6-31g(d,p)) followed by frequency analysis, solvent = THF
        self.name += '_fukui'
        name_short = str(os.path.basename(self.name))
        nw_input = self.name + ".nw"
        input_open = open(nw_input, "w")
        input_open.write("start " + name_short + "\n")
        input_open.writelines(""" memory total 5000 mb
 scratch_dir /scratch
 title "optimisation_vacuum"

charge """ + str(charge) + """
geometry noautosym
""")
        for j in self.atoms:
            input_open.write(j.label + " " + str(j.position[0]) + " " + str(j.position[1]) + " " + str(j.position[2]) + "\n")
        input_open.writelines("""end
 driver
  loose
  maxiter 30
 end
 dft
  xc b3lyp
  direct
  iterations 100
  grid xfine
  end
 basis
   * library 6-31g**
 end
 task dft optimize

 title "frequencies"
 task dft frequencies

  title "solvation_energy"
  cosmo
  do_cosmo_smd true
  solvent thf
  end
  dft
  odft
  fukui
  print "fukui information"
  mulliken
  print "mulliken ao"
  end
  task dft
""")
        input_open.close()
        self.nwchem_submit()

    def export_nwchem_initial(self, scratch, cluster):                # Quick NWChem optimisation (max number of steps is limited to 5)
        heavy_atoms_list = []
        light_atoms_list = []
        name_short = str(os.path.basename(self.name))
        nw_input = self.name + ".nw"
        input_open = open(nw_input, "w")
        input_open.write("start " + name_short + "\n")
        input_open.writelines(""" memory total 5000 mb
 scratch_dir """ + scratch + """
 title "optimisation_vacuum"

geometry noautosym
""")
        for j in self.atoms:
            input_open.write(j.label + " " + str(j.position[0]) + " " + str(j.position[1]) + " " + str(j.position[2]) + "\n")
            if j.label in heavy_atoms:
                if not j.label in heavy_atoms_list:
                    heavy_atoms_list.append(j.label)
            else:
                if not j.label in light_atoms_list:
                    light_atoms_list.append(j.label)
        input_open.writelines("""end
 driver
  loose
  maxiter 5
 end
 dft
  xc b3lyp
  iterations 100
  direct
  end
     """)
        if len(heavy_atoms_list) == 0:
            input_open.writelines("""basis spherical
* library 6-31g**
end
""")
        else:
            input_open.write("basis spherical\n")
            for ii in light_atoms_list:
                input_open.write(ii + ' library 6-31g**\n')
            for jj in heavy_atoms_list:
                input_open.write(jj + ' library crenbl_ecp\n')
            input_open.writelines("""end
ecp
""")
            for jj in heavy_atoms_list:
                input_open.write(jj + ' library crenbl_ecp\n')
            input_open.write('end\n')
        input_open.writelines("""
 task dft optimize



""")
        input_open.close()
        if cluster == "darwin":
            self.nwchem_submit_darwin()
        elif cluster == "local":
            self.nwchem_submit()
        elif cluster == 'darwin_liwei':
            self.nwchem_submit_liwei()

    def export_nwchem_initial_charge(self, charge, scratch, cluster): # Quick NWChem optimisation (max number of steps is limited to 5)
        heavy_atoms_list = []
        light_atoms_list = []
        name_short = str(os.path.basename(self.name))

        nw_input = self.name + ".nw"
        input_open = open(nw_input, "w")
        input_open.write("start " + name_short + "\n")
        input_open.writelines(""" memory total 5000 mb
 scratch_dir """ + scratch + """
 title "optimisation_vacuum"

charge """ + str(charge) + """
geometry noautosym
""")
        for j in self.atoms:
            input_open.write(
                j.label + " " + str(j.position[0]) + " " + str(j.position[1]) + " " + str(j.position[2]) + "\n")
            if j.label in heavy_atoms:
                if not j.label in heavy_atoms_list:
                    heavy_atoms_list.append(j.label)
            else:
                if not j.label in light_atoms_list:
                    light_atoms_list.append(j.label)
        input_open.writelines("""end
 driver
  loose
  maxiter 5
 end
 dft
  xc b3lyp
  iterations 300
  direct
  end
  """)
        if len(heavy_atoms_list) == 0:
            input_open.writelines("""basis spherical
* library 6-31g**
end
""")
        else:
            input_open.write("basis spherical\n")
            for ii in light_atoms_list:
                input_open.write(ii + ' library 6-31g**\n')
            for jj in heavy_atoms_list:
                input_open.write(jj + ' library crenbl_ecp\n')
            input_open.writelines("""end
ecp
""")
            for jj in heavy_atoms_list:
                input_open.write(jj + ' library crenbl_ecp\n')
            input_open.write('end\n')
        input_open.writelines("""
 task dft optimize


""")
        input_open.close()
        if cluster == "darwin":
            self.nwchem_submit_darwin()
        elif cluster == "local":
            self.nwchem_submit()
        elif cluster == 'darwin_liwei':
            self.nwchem_submit_liwei()

    def frequency_redo_submit(self, charge, scratch, cluster):
        heavy_atoms_list = []
        light_atoms_list = []
        input_resubmit = self.name + ".nw"
        name_short = os.path.basename(self.name)
        fopen = open(input_resubmit, "w")
        fopen.write("start " + name_short + "\n")
        fopen.writelines(""" memory total 5000 mb
 scratch_dir """ + scratch + """
 title "optimisation_vacuum"

charge """ + str(charge) + """
geometry noautosym
""")
        for j in self.atoms:
            fopen.write(j.label + " " + str(j.position[0]) + " " + str(j.position[1]) + " " + str(j.position[2]) + "\n")
            if j.label in heavy_atoms:
                if not j.label in heavy_atoms_list:
                    heavy_atoms_list.append(j.label)
            else:
                if not j.label in light_atoms_list:
                    light_atoms_list.append(j.label)
        fopen.writelines("""end
 driver
 maxiter 30
 end
 """)
        if len(heavy_atoms_list) == 0:
            fopen.writelines("""basis spherical
* library 6-31g**
end
""")
        else:
            fopen.write("basis spherical\n")
            for ii in light_atoms_list:
                fopen.write(ii + ' library 6-31g**\n')
            for jj in heavy_atoms_list:
                fopen.write(jj + ' library crenbl_ecp\n')
            fopen.writelines("""end
ecp
""")
            for jj in heavy_atoms_list:
                fopen.write(jj + ' library crenbl_ecp\n')
            fopen.write('end\n')
        fopen.writelines("""
 dft
  direct
  iterations 100
  xc b3lyp
  grid xfine
 end
 task dft optimize

  title "frequencies"
  task dft freq

  title "solvation_energy"
  cosmo
  do_cosmo_smd true
  solvent thf
  end
  task dft
""")
        fopen.close()
        if cluster == "darwin":
            self.nwchem_submit_darwin()
        elif cluster == "local":
            self.nwchem_submit()
        elif cluster == 'darwin_liwei':
            self.nwchem_submit_liwei()

    def load_checker(self):
        fl_output = self.name + ".out"
        if os.path.exists(fl_output):
            return "exists"
        else:
            return "no"

    def reaction_centre(self):
        name = self.name.split("/")
        name_short = name[-1]
        name_short = name_short.split("_")
        try:
            centre = name_short[2]
            self.centre = centre
        except:
            centre = 'N/A'
            self.centre = centre


    def connectivity_VDW(self):
        bonds = []
        for i in self.atoms:
            for j in self.atoms:
                if not j == i:
                    if i.check_connected(j):
                        bond = Bond(i,j,1)
                        bonds.append(bond)
        bonds_copy = []
        for k in bonds:
            for l in bonds:
                if not k == l:
                    if k.atom1 == l.atom2 and k.atom2 == l.atom1:
                        if not l in bonds_copy and not k in bonds_copy:
                            bonds_copy.append(l)
        for ii in bonds_copy:
            bonds.remove(ii)
        self.connectivity = bonds

    def connectivity_by_distance(self):
        processed_atoms = []
        bonds = []
        for i in self.atoms:
            processed_atoms.append(i)
            for j in self.atoms:
                if not j in processed_atoms:
                    label = 'N/A'
                    label1 = i.label + j.label
                    label2 = j.label + i.label
                    if label1 in Connectivity:
                        label = label1
                    if label2 in Connectivity:
                        label = label2
                    if not label == 'N/A':
                        if len(Connectivity[label]) == 1:
                            if i.distance(j) < Connectivity[label][0]:
                                bond = Bond(i, j, 1)
                                bonds.append(bond)
                        elif len(Connectivity[label]) == 4:
                            if i.distance(j) < Connectivity[label][0]:
                                if is_number(Connectivity[label][1]):
                                    if i.distance(j) > Connectivity[label][1]:
                                        bond = Bond(i, j, 1)
                                        bonds.append(bond)
                                    elif i.distance(j) < Connectivity[label][1]:
                                        if i.distance(j) > Connectivity[label][2]:
                                            bond = Bond(i, j, -1)
                                            i.aromaticity = True
                                            j.aromaticity = True
                                            bonds.append(bond)
                                        elif i.distance(j) < Connectivity[label][2]:
                                            if is_number(Connectivity[label][3]):
                                                if i.distance(j) < Connectivity[label][3]:
                                                    bond = Bond(i, j, 3)
                                                    bonds.append(bond)
                                                elif i.distance(j) > Connectivity[label][3]:
                                                    bond = Bond(i, j, 2)
                                                    bonds.append(bond)
                                            else :
                                                bond = Bond(i, j, 2)
                                                bonds.append(bond)
                                else:
                                    if i.distance(j) < Connectivity[label][2]:
                                        if is_number(Connectivity[label][3]):
                                            if i.distance(j) < Connectivity[label][3]:
                                                bond = Bond(i, j, 3)
                                                bonds.append(bond)
                                            elif i.distance(j) > Connectivity[label][3]:
                                                bond = Bond(i, j, 2)
                                                bonds.append(bond)
                                    elif i.distance(j) > Connectivity[label][2]:
                                        bond = Bond(i, j, 1)
                                        bonds.append(bond)
                        if label in Coordinative_bonds:
                            if i.distance(j) < Coordinative_bonds[label]:
                                if i.distance(j) > Connectivity[label][0]:
                                    bond = Bond(i, j, -2)
                                    bonds.append(bond)
        self.connectivity = bonds

    def radius_SMD_read(self):
        fileoutput = self.name + '.out'
        f = open(fileoutput, 'r')
        check1 = 0
        for line in f.readlines():
            if check1 == 1:
                line = line.split()
                if len(line) == 3:
                    for k in self.atoms:
                        if k.number == int(line[0]):
                            k.radius = float(line[2])
                elif len(line) == 0:
                    check1 = 0
            elif 'atomic radii =' in line:
                check1 = 1

    def connectivity_sdf(self):                                   # Function that reads bond information from the SDF MOL file; bond multiplicity is included
        filesdf = self.name + 'sdf'
        fopen = open(filesdf, 'r')
        check1 = 1
        bonds = []
        for line in fopen.readlines():
            if 'END' in line:                                            # Bond array finishes
                check1 = 0
            elif len(line.split()) == 3 and is_number(line.split()[0]) and check1 == 1:         # There is no keyword when bond array starts, so we compare a few properties of lines from the MOL2 files: amount of objects must be 3; first object must be a number
                lineproc = line.split()
                atom1 = self.determine_atom_by_number(int(lineproc[0]))
                atom2 = self.determine_atom_by_number(int(lineproc[1]))
                bond = Bond(atom1, atom2, lineproc[2])
                bonds.append(bond)
        fopen.close()
        return bonds

    def connectivity_mol2(self):
        filename = self.name + '.mol2'
        f = open(filename, "r")
        checkln = 0
        bondlst = []
        for line in f:
            if "@<TRIPOS>SUBSTRUCTURE" in line:
                break
            elif checkln == 1:
                lineproc = line.split()
                atom1 = self.determine_atom_by_number(int(lineproc[1]))
                atom2 = self.determine_atom_by_number(int(lineproc[2]))
                bond = Bond(atom1, atom2, str(lineproc[3]))
                bondlst.append(bond)
            elif "@<TRIPOS>BOND" in line:
                checkln += 1
        f.close()
        self.connectivity = bondlst

    def initial_rename(self):
        self.name += '_initial'

    def determine_atom_by_identity(self,identity):      # Function that finds the atom object using its identity (element + number; see above)
        result = "N/A"
        for i in self.atoms:
            if i.identity == identity:
                result = i
        return result

    def determine_atom_by_number(self,number):          # Function that finds the atom object using its sequence number
        result = "N/A"
        for i in self.atoms:
            if i.number == number:
                result = i
        return result


    def Pd_ipso_complex(self, atom_to_add, atom1, atom2, atom_added, H_added, substituent_H, distance):
        intermediates = []
        atoms_to_add1 = []
        atoms_to_add2 = []
        ipso_complex1 = copy.deepcopy(self)
        ipso_complex2 = copy.deepcopy(self)
        hydrogen = atom_to_add.Hattached[0]
        hydrogen1 = ipso_complex1.determine_atom_by_number(hydrogen.number)
        hydrogen2 = ipso_complex2.determine_atom_by_number(hydrogen.number)
        vector1 = vector_bond(atom_to_add, atom2)
        vector2 = vector_bond(atom_to_add, atom1)
        vector3 = vector_mult(vector1, vector2)
        vector4 = vector_bond(atom_to_add, hydrogen)
        vector_Pd1_elementary = vector_elementary(vector_add(vector_elementary(vector3), vector_elementary(vector4)))
        vector_Pd2_elementary = vector_elementary(vector_subtract(vector_elementary(vector4), vector_elementary(vector3)))
        carbon_Pd_length = 2.16
        carbon_hydrogen_length = 1.10
        vector_Pd1 = vector_multiply_by_scalar(vector_Pd1_elementary, carbon_Pd_length)
        vector_H1 = vector_multiply_by_scalar(vector_Pd2_elementary, carbon_hydrogen_length)
        vector_Pd2 = vector_multiply_by_scalar(vector_Pd2_elementary, carbon_Pd_length)
        vector_H2 = vector_multiply_by_scalar(vector_Pd1_elementary, carbon_hydrogen_length)
        hydrogen1.position = vector_add(atom_to_add.position, vector_H1)
        hydrogen2.position = vector_add(atom_to_add.position, vector_H2)
        Pd_position1 = vector_add(atom_to_add.position, vector_Pd1)
        Pd_position2 = vector_add(atom_to_add.position, vector_Pd2)
        Pd1 = Atom('Pd', Pd_position1)
        atoms_to_add1.append(Pd1)
        Pd2 = Atom('Pd', Pd_position2)
        atoms_to_add2.append(Pd2)
        ipso_complex1 = ipso_complex1.add_atoms(atoms_to_add1)
        ipso_complex1 = ipso_complex1.add_substituent(substituent_H, atom_to_add, Pd1, atom_added, H_added, atom1, atom2, distance)
        ipso_complex1.name = self.name + '_' + atom_to_add.label + str(atom_to_add.number) + '_1'
        intermediates.append(ipso_complex1)
        ipso_complex2 = ipso_complex2.add_atoms(atoms_to_add2)
        ipso_complex2 = ipso_complex2.add_substituent(substituent_H, atom_to_add, Pd2, atom_added, H_added, atom1, atom2, distance)
        ipso_complex2.name = self.name + '_' + atom_to_add.label + str(atom_to_add.number) + '_2'
        intermediates.append(ipso_complex2)
        return intermediates

    def add_substituent(self, substituent_H, atom_to_add, H_to_add, atom_added, H_added, atom_1, atom_2, distance):
        vector1 = vector_bond(atom_to_add, H_to_add)
        bond_length = distance
        vector_added = vector_multiply_by_scalar(vector_elementary(vector1),bond_length)
        new_position = vector_add(atom_to_add.position, vector_added)
        structure1 = self.remove_atoms([H_to_add])
        structure2 = substituent_H
        offset = vector_subtract(new_position, atom_added.position)
        for i in structure2.atoms:
            i.position = vector_add(i.position, offset)
        vector2 = vector_bond(atom_added, H_added)
        vector3 = vector_bond(atom_added, atom_to_add)
        axis = vector_mult(vector2, vector3)
        angle_rotate = angle(vector2, vector3)
        for i in structure2.atoms:
            i_offset = vector_subtract(i.position, atom_added.position)
            i_offset = vector_rotated(i_offset, axis, angle_rotate)
            i.position = vector_add(atom_added.position, i_offset)
            if not i == H_added:
                structure1 = structure1.add_atoms([i])
        structure1.name = self.name + '_' + atom_to_add.label + str(atom_to_add.number)
        return structure1


class StructureArray:

    """
The StructureArray class defines a group of structure objects as a single object

The required augment:
    structures (list of Structure class instances)

The methods available:

    1) cutoff_overall(self)
    Updates the list of structures by removing ones with the energy above 0.016 Hartree (10 kcal)
    relative to the lowest energy in the list

    2) def cutoff_center(self, centre)
    Updates the list of structures by removing ones with energy above 0.003 Hartree (2 kcal),
    specific to the given centre (structure.reaction_centre)

    3) reaction_centre_list(self)
    Return the list of reaction centres from the list of structures

    4) def log_initial(self, step)
    Creates an initial log file, named using step parameter

    5) export_nwchem(self, charge, scratch, cluster)
       export_nwchem_initial(self, scratch, cluster)
       export_nwchem_initial_charge(self, charge, scratch, cluster)
       initial_rename(self)
    Auxilary functions applying methods analogous to the ones described for a single structure

    6) radical_anions(self)
    Updates names of all structures in the list to '_radicalanion'

    """
    def __init__(self, structures):
        self.structures = structures

    def cutoff_overall(self):
        energies = []
        intermediates_new = []
        for i in self.structures:
            energies.append(i.energy)
        minimum = min(energies)
        for j in self.structures:
            if j.energy - minimum < 0.016:
                intermediates_new.append(j)
        self.structures = intermediates_new

    def cutoff_center(self, centre):
        energies = []
        intermediates_new = []
        for i in self.structures:
            if i.reaction_centre == centre:
                energies.append(i.energy)
        minimum = min(energies)
        for j in self.structures:
            if j.reaction_centre == centre:
                if j.energy - minimum < 0.003:
                    intermediates_new.append(j)
        intermediates_new_list = StructureArray(intermediates_new)
        return intermediates_new_list

    def reaction_centre_list(self):
        centre_list = []
        for i in self.structures:
            if i.reaction_centre not in centre_list:
                centre_list.append(i.reaction_centre)
        return centre_list

    def log_initial(self, step):
        path = os.path.dirname(self.structures[0].name)
        log_name = path + '/' + step + ".log"
        f = open(log_name, 'w')
        for i in self.structures:
            f.write(i.name + "  submitted\n")
        f.close()

    def export_nwchem(self, charge, scratch, cluster):
        for i in self.structures:
            i.export_nwchem(charge, scratch, cluster)

    def radical_anions(self):
        new_list = copy.deepcopy(self)
        for i in new_list.structures:
            i.name += '_radicalanion'
        return new_list

    def export_nwchem_initial(self, scratch, cluster):
        for i in self.structures:
            i.export_nwchem_initial(scratch, cluster)

    def export_nwchem_initial_charge(self, charge, scratch, cluster):
        for i in self.structures:
            i.export_nwchem_initial_charge(charge, scratch, cluster)

    def initial_rename(self):
        for i in self.structures:
            i.initial_rename()


class Output:

    """
Output class associated with the computational output and log files. The class contains file processing methods

Required parameters:
    filename

Available methods:

    1) keyline_coordinates_read(self, line)
    Auxilary function returning a list of atom instances by parsing the output file and using 'line' as a flag augument

    2) reaction_centre(self)
    Auxilary function returning the reaction centre by parsing the filename if it adopts the format
    {Experiment_ID}_{conformer_ID}_{reaction_centre}_{sequential_number}

    3) saddle_fixer(self, charge, scratch, cluster)
    Function that generates and submits two calculation alongside the vibrational mode for a given imaginary frequence
    using charge, scratch directory and cluster options as parameters

    4) final_energy_nwchem(self)
    Returns final uncorrected energy by parsing NWChem output file

    5) energy_correction(self)
    Returns zero-point energy correction by parsing NWChem output file

    6) normal_termination_nwchem(self)
    Return 'check' flag if the computation has been completed successfully

    7) frequency_analysis(self)
    Returns the outcome of frequency analysis: 'minimum' for a local minimum, 'saddle' for a saddle point on PES,
    'error' for a uncorrectly terminated calculation

    8) corrected_energy(self)
    Returns coorected energy (uncorrected energy + zero-point energy correction)

    9) error_check(self)
    Returns different type of error following the geometry optimisation

    10) result_extraction(self, molecule, scratch)
    Return Result class instance, using molecule (structure instance) and scratch as additional parameters

    11) intermediate_output_nwchem(self)
    Return structure instance and updates its' properties from the computational output

    12) minimal_energy_molecule_list(self, charge, scratch, cluster)
    Returns the minimal energy of all the structures in the given log file

    13) step_finished(self)
    Auxilary function returning a bool flag whether the step corresponding to the given log file is complete

    14) intermediates_result_builder(self, molecule, charge, scratch, cluster)
    Return the list of Result instances for all the calulations in the given log file

    """

    def __init__(self, filename):
        self.filename = filename

    def keyline_coordinates_read(self, line):
        f = open(self.filename, 'r')
        atoms = []
        check1 = 0
        for l in f.readlines():
            if check1 == 2:
                l = l.split()
                if len(l) > 1:
                    atom = Atom(l[1],[l[3], l[4], l[5]])
                    atoms.append(atom)
                else:
                    check1 = 0
            elif check1 == 1:
                check1 += 1
            elif line in l:
                check1 = 1
                atoms = []
        f.close()
        return atoms

    def reaction_centre(self):
        name = self.filename.split("/")
        name_short = name[-1]
        name_short = name_short.split("_")
        try:
            centre = name_short[2]
            self.centre = centre
        except:
            centre = 'N/A'
            self.centre = centre

    def saddle_fixer(self, charge, scratch, cluster):
        atoms_positive = self.keyline_coordinates_read("Geometry after  100.0%")
        positive_intermediate = Structure(atoms_positive)
        positive_intermediate.name = str(self.filename)[:-4] + "_redo1"
        positive_intermediate.frequency_redo_submit(charge, scratch, cluster)
        atoms_negative = self.keyline_coordinates_read("Geometry after -100.0%")
        negative_intermediate = Structure(atoms_negative)
        negative_intermediate.name = str(self.filename)[:-4] + "_redo2"
        negative_intermediate.frequency_redo_submit(charge, scratch, cluster)

    def final_energy_nwchem(self):  # final energy of the nwchem calculation
        f = open(self.filename, "r")
        energy = "N/A"
        for line in f.readlines():
            if "Total DFT energy" in line:
                line = line.split()
                pos = line.index("=") + 1
                energy = float(line[pos])
        f.close()
        return energy

    def energy_correction(self):  # Zero point correction energy in NWchem
        f = open(self.filename, "r")
        energy_correction = "N/A"
        for line in f.readlines():
            if "Zero-Point correction to Energy" in line:
                line = line.split()
                pos = line.index("=") + 1
                energy_correction = float(line[pos]) / 627.503
        f.close()
        return energy_correction

    def normal_termination_nwchem(self):  # normal termination check for nwchem
        f = open(self.filename, "r")
        lines = f.readlines()
        check = False
        if "Total times  cpu:" in lines[-1]:
            check = True
        f.close()
        return check

    def frequency_analysis(self):
        if self.normal_termination_nwchem():
            f = open(self.filename, 'r')
            frequencies = []
            for line in f.readlines():
                if "P.Frequency" in line:
                    line = line.split()
                    for i in line:
                        if is_number(i):
                            frequencies.append(float(i))
            minimal = min(frequencies)
            f.close()
            if minimal > -0.5:
                return "minimum"
            else:
                return "saddle"

        else:
            return "error"

    def corrected_energy(self):
        e1 = self.final_energy_nwchem()
        e2 = self.energy_correction()
        return e1 + e2

    def error_check(self):
        with open(self.filename) as f:
            for line in f.readlines():
                if "Failed to converge in maximum number of steps or available time" in line:
                    return "steps_exceeded"
        with open(self.filename) as f1:
            for line in f1.readlines():
                if 'driver: task_gradient failed' in line:
                    return 'bad_geometry'
        raise NameError('error')

    def result_extraction(self, molecule, scratch):
        intermediate = self.intermediate_output_nwchem()
        intermediate.radius_SMD_read()
        intermediate.connectivity_by_distance()
        intermediate.visualisation()
        result = Result(molecule, intermediate.centre)
        result.intermediate = intermediate
        if not intermediate.Ecorr == 'N/A':
            result.energy = intermediate.energy + intermediate.Ecorr
            result.level = "corrected"
        else:
            result.energy = intermediate.energy
            result.level = "uncorrected"
        scratch_clean(self.filename, scratch)
        result.result_resubmitted()
        return result

    def intermediate_output_nwchem(self):
        f = open(self.filename, "r")
        check1 = 0
        check2 = 0
        atmlst = []
        for line in f.readlines():
            if 'Geometry "geometry" -> "geometry"' in line:
                check1 = 1
                atmlst = []
                check2 = 0
                continue
            elif check1 == 1:
                if "---" in line:
                    check2 += 1
                    continue
                elif check2 == 2:
                    if len(line.split()) == 6:
                        line = line.split()
                        atom = Atom(line[1], [line[3], line[4], line[5]])
                        atmlst.append(atom)
                        if atom.label == "H":  # choosing atom type
                            atom.type = "light"
                        else:
                            atom.type = "heavy"
                        atom.number = int(line[0])
                        continue
                    else:
                        check1 = 0
                        continue
            else:
                continue
        f.close()
        intermediate_out = Structure(atmlst)
        intermediate_out.name = str(self.filename)[:-4]
        intermediate_out.reaction_centre()
        intermediate_out.energy = self.final_energy_nwchem()
        intermediate_out.Ecorr = self.energy_correction()
        intermediate_out.charge = 0
        return intermediate_out

    def minimal_energy_molecule_list(self, charge, scratch, cluster):
        load_log_data = Loader(self.filename)
        log_data = load_log_data.log_data_loader()
        data_processed = log_data.log_processing(charge, scratch, cluster)
        dict = data_processed.data.copy()
        energy = []
        for i in dict:
            if dict[i] == "complete":
                floutput = Output(i + '.out')
                energy.append(floutput.corrected_energy())
        return min(energy)

    def step_finished(self):
        try:
            f = open(self.filename, 'r')
            for line in f.readlines():
                if line.split()[-1] == "finished":
                    return True
            f.close()
        except:
            return False

    def intermediates_result_builder(self, molecule, charge, scratch, cluster):
        log_data_upload = Loader(self.filename)
        log_data = log_data_upload.log_data_loader()
        log_data = log_data.log_processing(charge, scratch, cluster)
        results = []
        result_list = None
        dict = log_data.data.copy()
        for i in dict:
            if dict[i] == "complete":
                floutput = Output(i + '.out')
                result = floutput.result_extraction(molecule, scratch)
                results.append(result)
        if len(results) > 0:
            result_list = ResultArray(results)
            result_list.processing_results()
        return result_list


class Result:

    """
Result class instance that is used to summarise computational results

Required augments:
    molecule - structure instance that was computed
    centre - reaction centre in the structure the results correspond to

Methods available:

    1) result_resubmitted(self)
    Bool flag to distinguish redrawn intermediates

    2) highest_conformer(self, result2)
    Function returning the highest of two - self and result2 - energy conformers

    """

    def __init__(self, molecule, centre):
        self.molecule = molecule
        self.centre = centre
        self.energy = 'N/A'
        self.level = 'N/A'
        self.intermediate = 'N/A'
        self.resubmitted = False

    def result_resubmitted(self):
        lst = self.intermediate.name.split('_')
        try:
            if 'redrawn1' in lst or 'redrawn2' in lst:
                self.resubmitted = True
            else:
                self.resubmitted = False
        except IndexError:
            self.resubmitted = False

    def highest_conformer(self, result2):  # choosing the most stable conformer
        if self.centre == result2.centre:
            if self.energy < result2.energy:
                return result2
            else:
                return self


class ResultArray:

    """
ResultArray class instance that is comprised of multiple computational results to describe multiple structures calculations

Required augments:
    results - list of Result instances to be included

Methods available:

    1) processing_energies(self)
    Offset all the energies in the list by the minimum energy in that list

    2) sorting_by_energy(self)
    Sort the results in the list by energy (ascending)

    3) minimal_intermediate(self)
    Return the minimal energy of the results in the list

    4) processing_results(self)
    Processing function removing all the higher energy results from the list (except for manualy redrawn intermediates)

    5) results_print(self, dirname, difference)
    Function generating text output calculating relative energies for all the results in the list, using dirname and
    difference (minimal energy of the lowest intermediate relative to the chosen reference) as additional
    parameters

    """

    def __init__(self, results):
        self.results = results

    def processing_energies(self):  # calibrating energies
        minimum = self.minimal_intermediate()
        for j in self.results:
            j.energy = ((j.energy - minimum) * 627.503)

    def sorting_by_energy(self):
        try:
            self.results.sort(key=lambda x: x.energy, reverse=False)
        except:
            pass

    def minimal_intermediate(self):
        enrglst = []
        for i in self.results:
            enrglst.append(i.energy)
        minimum = min(enrglst)
        return minimum

    def processing_results(self):
        results_remove = []
        for i in self.results:
            for j in self.results:
                if not i == j:
                    if i.centre == j.centre:
                        if not i.resubmitted and not j.resubmitted:
                            results_remove.append(i.highest_conformer(j))
        for k in results_remove:
            if k in self.results:
                self.results.remove(k)
        self.processing_energies()

    def results_print(self, dirname, difference):
        flname = dirname + '/result.txt'
        f = open(flname, 'w')
        f.write('Position   Energy of the intermediate   Filename\n')
        self.sorting_by_energy()
        jj = 1
        for i in self.results:
            if not i.energy == 'N/A':
                if i.resubmitted:
                    data = ['position' + str(jj) + '(resubmitted)', str(round(i.energy, 1)), os.path.basename(i.intermediate.name)]
                else:
                    data = ['position' + str(jj), str(round(i.energy, 1)), os.path.basename(i.intermediate.name)]
            else:
                data = ['position' + str(jj), 'N/A', os.path.basename(i.intermediate.name)]
            f.write('{0[0]:<20}{0[1]:<7}{0[2]:>28}'.format(data) + "\n")
            jj += 1
        f.write('Deprotonation Energy Rank  ' + str(difference) + '\n')
        f.close()


class Log_data:

    """
Log_Data class is used to store computational status data for multiple computations of different stages

Required augments:
    data - dictionary in the format computation_name : status

Methods available:

    1) saddle_loader(self, charge, scratch, cluster)
    Analyses and updates log data adding required saddle-point correction calculations

    2) error_loader(self, charge, scratch, cluster)
    Analyses and updates log data adding required information about erroneous calculations

    3) step_finished(self)
    Bool flag to verify whether all the calculations related to the log data are complete

    4) log_processing(self, charge, scratch, cluster)
    Analyses and updates the normal processing of calculations

    5) initial_finished(self)
    Bool flag to verify whether all the calculations related to the log data are complete; relying on pbs queueing
    system log files

    6) initial_processing_darwin(self)
    Analyses and updates the normal processing of initial step calculations

    7) initial_analysis(self, charge, scratch, cluster)
    Analyses and moves computational processes to the next step from initial; needs additional parameters to submit
    further calculations (charge, scratch, cluster)

    8) log_data_export(self, name)
    Exports log data into the log txt file using name of the step as an additional parameter

    9) minimal_energy_structure(self)
    Return the minimal energy intermediate from the log data

    """

    def __init__(self, data):
        self.data = data
        self.finished = False
        self.name = 'N/A'

    def saddle_loader(self, charge, scratch, cluster):
        log_data_new = {}
        dict = self.data.copy()
        for i in dict:
            if dict[i] == 'saddle':
                check_submit = 'yes'
                energies_to_compare = []
                floutput = Output(i + '.out')
                intermediate = floutput.intermediate_output_nwchem()
                for j in dict:
                    if dict[j] == 'complete':
                        floutput2 = Output(j + '.out')
                        energy_check = floutput2.final_energy_nwchem()
                        energies_to_compare.append(energy_check)
                        if floutput2.reaction_centre == intermediate.reaction_centre:
                            if intermediate.energy - energy_check > 0.003:
                                log_data_new[i] = 'high_energy_saddle'
                                check_submit = 'no'
                                break
                try:
                    minimum_energy = min(energies_to_compare)
                    if check_submit == 'yes':
                        if intermediate.energy - minimum_energy > 0.020:
                            log_data_new[i] = 'high_energy_saddle'
                            check_submit = 'no'
                except:
                    check_submit = 'yes'

                if check_submit == 'yes':
                    flname_short = os.path.basename(i)
                    split = flname_short.split('_')
                    check = 0
                    for ii in split:
                        if ii == 'redo1' or ii == 'redo2' or ii == 'redo':
                            check += 1
                    if check == 3:
                        log_data_new[i] = 'processed'
                        check_submit = 'no'

                if check_submit == 'yes':
                    floutput.saddle_fixer(charge, scratch, cluster)
                    log_data_new[i] = 'processed'
                    log_data_new[i + '_redo1'] = 'submitted'
                    log_data_new[i + '_redo2'] = 'submitted'
            else:
                log_data_new[i] = dict[i]
        self.data = log_data_new

    def error_loader(self, charge, scratch, cluster):
        log_data_new = {}
        dict = self.data.copy()
        for i in dict:
            if dict[i] == 'steps_exceeded':
                floutput = Output(i + '.out')
                intermediate = floutput.intermediate_output_nwchem()
                intermediate.name = intermediate.name + '_redo'
                intermediate.frequency_redo_submit(charge, scratch, cluster)
                log_data_new[i] = 'processed'
                log_data_new[i + '_redo'] = 'submitted'
            elif dict[i] == 'bad_geometry':
                log_data_new[i] = 'high_energy_saddle'
            else:
                log_data_new[i] = dict[i]
        self.data = log_data_new

    def step_finished(self):
        dict = self.data.copy()
        for i in dict:
            if dict[i] == 'submitted' or dict[i] == 'processing':
                return False
            else:
                continue
        return True

    def log_processing(self, charge, scratch, cluster):
        log_data_new = {}
        dict = self.data.copy()
        for i in dict:
            floutput = Output(i + '.out')
            if os.path.isfile(floutput.filename):
                if dict[i] == 'submitted' or dict[i] == 'processing':
                    if floutput.normal_termination_nwchem():
                        if floutput.frequency_analysis() == "minimum":
                            log_data_new[i] = 'complete'
                        elif floutput.frequency_analysis() == "saddle":
                            log_data_new[i] = 'saddle'
                    else:
                        try:
                            log_data_new[i] = str(floutput.error_check())
                        except:
                            log_data_new[i] = 'processing'
                else:
                    log_data_new[i] = dict[i]
            else:
                log_data_new[i] = 'submitted'
        log_data = Log_data(log_data_new)
        log_data.saddle_loader(charge, scratch, cluster)
        log_data.error_loader(charge, scratch, cluster)
        if log_data.step_finished():
            log_data.finished = True
        return log_data

    def initial_finished(self):
        dict = self.data.copy()
        for i in dict:
            if not os.path.isfile(i + '.outpbs'):
                    return False
        return True

    def initial_processing_darwin(self):
        log_data_new = {}
        dict = self.data.copy()
        for i in dict:
            floutput = Output(i + '.out')
            if os.path.isfile(floutput.filename):
                if dict[i] == 'submitted' or dict[i] == 'processing':
                    if floutput.normal_termination_nwchem():
                        log_data_new[i] = 'complete'
                    else:
                        try:
                            log_data_new[i] = str(floutput.error_check())
                        except:
                            log_data_new[i] = 'processing'
                else:
                    log_data_new[i] = dict[i]
            else:
                log_data_new[i] = 'submitted'
        log_data = Log_data(log_data_new)
        if log_data.step_finished():
            log_data.finished = True
        return log_data

    def initial_analysis(self, charge, scratch, cluster):
        intermediates = []
        dict = self.data.copy()
        for i in dict:
            floutput = Output(i + '.out')
            intermediate = floutput.intermediate_output_nwchem()
            intermediate.name = str(intermediate.name)[:-8]
            intermediates.append(intermediate)
        intermediate_list = StructureArray(intermediates)
        intermediate_list.cutoff_overall()
        centre_list = intermediate_list.reaction_centre_list()
        intermediates_new = []
        for j in centre_list:
            intermediates_to_add = intermediate_list.cutoff_center(j)
            for jj in intermediates_to_add.structures:
                intermediates_new.append(jj)
        intermediate_new_list = StructureArray(intermediates_new)
        intermediate_new_list.export_nwchem(charge, scratch, cluster)
        intermediate_new_list.log_initial('step2')

    def log_data_export(self, name):
        dict = self.data.copy()
        f = open(name, 'w')
        for i in self.data:
            f.write(i + '   ' + dict[i] + '\n')
        if self.finished:
            f.write('step   finished\n')
        f.close()

    def minimal_energy_structure(self):
        dict = self.data.copy()
        minimal_intermediate = 'N/A'
        minimal_energy = 99999
        for i in dict:
            if dict[i] == 'complete':
                output = Output(i + '.out')
                intermediate = output.intermediate_output_nwchem()
                energy = float(intermediate.energy) + float(intermediate.Ecorr)
                if energy < minimal_energy:
                    minimal_intermediate = intermediate
        return minimal_intermediate


class Loader:

    """

Loader class combines methods to parse files extracting data

Required parameters:
    source_filename - filename to be parsed

Available methods:
    1) molecule(self)
    Returns a structure object from an SDF file

    2) molecule_mol2(self)
    Returns a structure object from MOL2 file

    3) molecule_xyz(self)
    Returns a structure object from XYZ file

    4) structure_redrawn(self, charge, scratch, cluster)
    Extracts the redrawn structure and submits NWChem calculation using additional paramters
    (charge, scratch, cluster)

    5) initial_loader(self)
    Reads multiconformational SDF input file and submits initial computational step

    6) initial_loader_darwin(self)
    Reads multiconformational SDF input file and submits initial computational step on CPU cluster

    7) log_data_loader(self)
    Loads log data from log txt file

    8) step_finished(self)
    Checks if the computational step is finished and updates the log txt file


    """

    def __init__(self, source_filename):
        self.source_filename = source_filename

    def molecule(self):  # creating a molecule object from an sdf file
        f = open(self.source_filename, "r")
        atmnumb = 1
        atmlst = []
        check1 = 1
        for line in f:
            if 'END' in line:
                check1 = 0
            elif '.' in line and check1 == 1 and is_number(line.split()[0]):
                lineproc = line.split()
                lineproc[1] = label_format(lineproc[1])
                atom = Atom(lineproc[3], [lineproc[0], lineproc[1], lineproc[2]])
                atom.connectivity_number = atom_string.index(atom.label)
                atmlst.append(atom)
                if atom.label == "H":  # choosing atom type
                    atom.type = "light"
                else:
                    atom.type = "heavy"
                atom.number = atmnumb
                atmnumb += 1
        f.close()
        molecule = Structure(atmlst)
        molecule.name = str(self.source_filename)[:-4]
        molecule.charge = 0
        molecule.solvent = "thf"
        molecule.Hconnected()
        return molecule

    def molecule_mol2(self):
        f = open(self.source_filename, "r")
        checkln = 0
        atmnumb = 1
        atmlst = []
        for line in f:
            if "@<TRIPOS>BOND" in line:
                break
            elif checkln == 1:
                lineproc = line.split()
                lineproc[1] = strip_numbers(lineproc[1])
                lineproc[1] = label_format(lineproc[1])
                atom = Atom(lineproc[1], [lineproc[2], lineproc[3], lineproc[4]])
                atom.number = atmnumb
                check_aromatic = lineproc[5].split('.')
                try:
                    if check_aromatic[1] == 'ar':
                        atom.aromaticity = True
                    else :
                        atom.aromaticity = False
                except IndexError:
                    atom.aromaticity = False
                if atom.label == "H":  # choosing atom type
                    atom.type = "light"
                else:
                    atom.type = "heavy"
                atmlst.append(atom)
                atmnumb += 1
            elif "@<TRIPOS>ATOM" in line:
                checkln = 1
        f.close()
        molecule = Structure(atmlst)
        molecule.name = str(self.source_filename)[:-5]
        molecule.charge = 0
        molecule.Hconnected()
        molecule.connectivity_mol2()
        return molecule

    def molecule_xyz(self):                                                         # Creating a structure object from an xyz file; No bonding information is given
        f = open(self.source_filename, "r")
        checkln = 0
        atmnumb = 1
        atmlst = []
        for line in f:
            if checkln == 2:
                lineproc = line.split()
                atom = Atom(lineproc[0],[lineproc[1],lineproc[2],lineproc[3]])
                atmlst.append(atom)
                atom.number = atmnumb
                if atom.label == "H":  # choosing atom type
                    atom.type = "light"
                else:
                    atom.type = "heavy"
                atmnumb += 1
            else :
                checkln += 1
        f.close()
        atmlst = recenter(atmlst)
        molecule = Structure(atmlst)
        molecule.connectivity_by_distance()
        molecule.Hconnected()
        molecule.name = str(self.source_filename)[:-4]
        molecule.charge = 0
        return molecule

    def structure_redrawn(self, charge, scratch, cluster):
        atmlst = []
        flopen = open(self.source_filename, 'r')
        for line in flopen.readlines():
            line = line.split()
            atom = Atom(line[0], [line[1], line[2], line[3]])
            atmlst.append(atom)
        structure = Structure(atmlst)
        structure.name = str(self.source_filename)[:-6]
        structure.export_nwchem(charge, scratch, cluster)

    def initial_loader(self):                           # Initial loader of a molecule using a smiles file; molecular file *.sdf and conformer file *.conf must also be present in the folder
        fldbg = str(self.source_filename)[:-6] + "sdf"
        flconf = str(fldbg[:-3]) + "conf"
        loader_molecule = Loader(fldbg)
        molecule_build = loader_molecule.molecule()
        conf_list = molecule_build.conformers(flconf)
        conf_list = limit_list(conf_list)
        conformer_list = StructureArray(conf_list)
        conformer_list.export_nwchem(0, '/scratch', 'local')
        conformer_list.log_initial('step1')
        intermediates = conformer_list.Li_intermediates()
        intermediates.initial_rename()
        intermediates.export_nwchem_initial('/scratch', 'local')
        intermediates.log_initial('step11')
        dirname_archive = os.path.dirname(self.source_filename) + '/archive'
        os.system('mkdir ' + dirname_archive)
        flsmiles_short = os.path.basename(self.source_filename)
        os.system('mv ' + self.source_filename + ' ' + dirname_archive + '/' + flsmiles_short)
        os.system("ls > /dev/null")

    def initial_loader_darwin(self):
        fldbg = str(self.source_filename)[:-6] + "sdf"
        flconf = str(fldbg[:-3]) + "conf"
        loader_molecule = Loader(fldbg)
        molecule_build = loader_molecule.molecule()
        conf_list = molecule_build.conformers(flconf)
        conf_list = limit_list(conf_list)
        conformer_list = StructureArray(conf_list)
        conformer_list.export_nwchem(0, '/scratch/mk787', 'darwin')
        conformer_list.log_initial('step1')
        intermediates = conformer_list.Li_intermediates()
        intermediates.initial_rename()
        intermediates.export_nwchem_initial('/scratch/mk787', 'darwin')
        intermediates.log_initial('step11')
        dirname_archive = os.path.dirname(self.source_filename) + '/archive'
        os.system('mkdir ' + dirname_archive)
        flsmiles_short = os.path.basename(self.source_filename)
        os.system('mv ' + self.source_filename + ' ' + dirname_archive + '/' + flsmiles_short)
        os.system("ls > /dev/null")

    def log_data_loader(self):
        f = open(self.source_filename,'r')
        log_data = {}
        check_finished = 'no'
        for line in f:
            line = line.split()
            if not line[0] == 'step':
                log_data[line[0]] = line[1]
            elif line[0] == 'step':
                check_finished = 'yes'
                break
        log_data = Log_data(log_data)
        if check_finished == 'yes':
            log_data.finished = True
        return log_data

    def step_finished(self):
        try:
            log_data = self.log_data_loader()
            if log_data.step_finished():
                f = open(self.source_filename,'a')
                f.write('step   finished\n')
                f.close()
                return True
        except ValueError:
            return True

#######################################             Auxillary parsing functions                 ##############################################

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def strip_numbers(line):
    line_list = list(line)
    line_letters = []
    for j in line_list:
        if not is_number(j):
            line_letters.append(j)
    i = 0
    line_strip = ""
    while i < len(line_letters):
        line_strip += line_letters[i]
        i += 1
    return line_strip


def limit_list(list):
    new_list = []
    i = 0
    while i < 5:
        try:
            new_list.append(list[i])
            i += 1
        except IndexError:
            break
    return new_list

def limit_list_loose(list):
    new_list = []
    i = 0
    while i < 10:
        try:
            new_list.append(list[i])
            i += 1
        except IndexError:
            break
    return new_list


def label_format(label):
    s = list(label)
    if len(s) == 2:
        if s[1].isupper():
            s[1] = s[1].lower()
    label_new = ''
    for i in s:
        label_new += i
    return label_new


def brackets_separate(flname):
    f = open(flname, "r")
    fnew = str(flname) + "_new"
    fnewopen = open(fnew, "w")
    for line in f:
        if "][" in line:
            line = line.replace("][", "],[")
            fnewopen.write(line)
        else:
            fnewopen.write(line)
    f.close()
    fnewopen.close()
    os.system("mv " + fnew + " " + flname)
    os.system("ls > /dev/null")






##################################                  Auxillary Vector functions              #####################################


def vector_length(vector):
    x1 = vector[0]
    y1 = vector[1]
    z1 = vector[2]
    return math.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2)


def vector_bond(atom1, atom2):
    return [(atom2.position[0] - atom1.position[0]), (atom2.position[1] - atom1.position[1]),
            (atom2.position[2] - atom1.position[2])]


def vector_bond_elementary(atom1, atom2):
    return [vector_bond(atom1, atom2)[0] / vector_length(vector_bond(atom1, atom2)),
            vector_bond(atom1, atom2)[1] / vector_length(vector_bond(atom1, atom2)),
            vector_bond(atom1, atom2)[2] / vector_length(vector_bond(atom1, atom2))]


def vector_elementary(vector):
    return [vector[0] / vector_length(vector), vector[1] / vector_length(vector), vector[2] / vector_length(vector)]


def vector_multiply_by_scalar(vector, scalar):
    return [vector[0] * scalar, vector[1] * scalar, vector[2] * scalar]

def vector_add(vector1, vector2):
    return [(vector1[0] + vector2[0]), (vector1[1] + vector2[1]), (vector1[2] + vector2[2])]

def vector_subtract(vector1, vector2):
    return [(vector1[0] - vector2[0]), (vector1[1] - vector2[1]), (vector1[2] - vector2[2])]

def center(atom1, atom2):
    return [(atom1.position[0] + atom2.position[0]) / 2, (atom1.position[1] + atom2.position[1]) / 2,
            (atom1.position[2] + atom2.position[2]) / 2]


def vector_mult(vector1, vector2):
    x1 = vector1[0]
    y1 = vector1[1]
    z1 = vector1[2]
    x2 = vector2[0]
    y2 = vector2[1]
    z2 = vector2[2]
    return [(y1 * z2 - z1 * y2), (z1 * x2 - x1 * z2), (x1 * y2 - y1 * x2)]


def scalar_mult(vector1, vector2):
    x1 = vector1[0]
    y1 = vector1[1]
    z1 = vector1[2]
    x2 = vector2[0]
    y2 = vector2[1]
    z2 = vector2[2]
    return (x1 * x2 + y1 * y2 + z1 * z2)


def angle(vector1, vector2):
    return math.acos(scalar_mult(vector1, vector2) / (vector_length(vector1) * vector_length(vector2)))

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2)
    b, c, d = -axis*math.sin(theta/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def vector_rotated(vector, axis, theta):
    return np.dot(rotation_matrix(axis, theta), vector)


def quaternion(atom1, atom2):
    vector1 = [0, 1, 0]
    vector2 = vector_bond_elementary(atom1, atom2)
    return [vector_mult(vector1, vector2)[0], vector_mult(vector1, vector2)[1], vector_mult(vector1, vector2)[2],
            angle(vector1, vector2)]


def move_closer(atom1, atom2):
    vector = vector_bond_elementary(atom2, atom1)
    length = atom1.distance(atom2) * 0.8
    vector_new = vector_multiply_by_scalar(vector, length)
    return [(atom2.position[0] + vector_new[0]), (atom2.position[1] + vector_new[1]),
            (atom2.position[2] + vector_new[2])]

def frac_cartesian(fractional, alpha, beta, gamma, a, b, c):
    conv = conversion_matrix(alpha, beta, gamma, a, b, c)
    x = float(fractional[0])
    y = float(fractional[1])
    z = float(fractional[2])
    return [conv[0] * x + conv[1] * y + conv[2] * z, conv[3] * x + conv[4] * y + conv[5] * z, conv[6] * x + conv[7] * y + conv[8] * z]

def conversion_matrix(alpha, beta, gamma, a, b, c):
    return [a, b * math.cos(gamma), c * math.cos(beta), 0, b * math.sin(gamma), c * (math.cos(alpha) - math.cos(beta)*math.cos(gamma)) / math.sin(gamma), 0, 0, c * volume_unit(alpha, beta, gamma) / math.sin(gamma)]

def volume_unit(alpha, beta, gamma):
    return math.sqrt(1 - math.cos(alpha)**2 - math.cos(beta)**2 - math.cos(gamma)**2 + 2 * math.cos(alpha) * math.cos(beta) * math.cos(gamma))


def NO2_builder(carbon, vector1, vector2):
    vector3 = vector_elementary(vector_mult(vector1, vector2))
    vector4 = vector_elementary(vector_add(vector3, vector_elementary(vector1)))
    vector5 = vector_elementary(vector_subtract(vector_elementary(vector1), vector3))
    carbon_nitrogen_length = 1.5
    nitrogen_oxygen_length = 1.22





#############################################          Auxillary functions         #################################################

def strip_brackets(line):
    line_list = list(line)
    line_numbers = []
    i = 0
    line_strip = ""
    for j in line_list:
        if not j == "(":
            line_numbers.append(j)
        else :
            break
    while i < len(line_numbers):
        line_strip += line_numbers[i]
        i +=1
    return line_strip
















        
    

        




    


