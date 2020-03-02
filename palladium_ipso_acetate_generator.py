#! /usr/bin/python
import sys
import os
import math
import copy
import glob
import numpy as np
import liwei


"""

The first automated script in the electrophilic addition mechanism (Pd(OAc)+ added as an electrophile)
that submits step1 (molecule geometry optimisation) and step11 (initial geometry optimisation for ipso-Pd intermediates)
taking $MOLECULE.smiles, $MOLECULE.mol2, $MOLECULE.sdf, $MOLECULE.conf
(all molecular conformations in SDF format) and Pd_OAc_acetate2.xyz as required input files.

The script operates in a given working directory and requires cluster to be configured beforehand

"""

cluster = 'darwin_liwei'
scratch = '/scratch/lc640'
cur_dir = sys.argv[1]

for flname in glob.glob(cur_dir + "/*.smiles"):
    flsdf = str(flname)[:-6] + "sdf"
    flmol2 = str(flname)[:-6] + 'mol2'
    flconf = str(flsdf[:-3]) + "conf"
    substituent_xyz = liwei.Loader('/home/lc640/python_trial/Pd_OAc.xyz')
    substituent_H = substituent_xyz.molecule_xyz()
    atom_added = substituent_H.determine_atom_by_number(1)
    H_added = substituent_H.determine_atom_by_number(9)
    molecule_xyz = liwei.Loader('/home/lc640/python_trial/benzene.xyz')
    loader_molecule = liwei.Loader(flmol2)
    molecule= loader_molecule.molecule_mol2()
    conf_list = molecule.conformers(flconf)
    conf_list = liwei.limit_list(conf_list)
    conformer_list = liwei.StructureArray(conf_list)
    conformer_list.export_nwchem(0, scratch, cluster)
    conformer_list.log_initial('step1')
    intermediates = []
    for k in conformer_list.structures:
        for i in k.atoms:
            if i.label == 'C':
                connected_atoms = k.connected_to_atom(i)
                if len(connected_atoms) == 2:
                    if len(i.Hattached) == 1:
                        atom1 = connected_atoms[0]
                        atom2 = connected_atoms[1]
                        atom_to_add = i
                        H_to_add = i.Hattached[0]
                        intermediate_to_submit = k.Pd_ipso_complex(atom_to_add, atom1, atom2, atom_added, H_added, substituent_H, 1.94)
                        for j in intermediate_to_submit:
                            j_copy = copy.deepcopy(j)
                            intermediates.append(j_copy)
    intermediates = liwei.StructureArray(intermediates)
    intermediates.initial_rename()
    intermediates.export_nwchem_initial_charge(1, scratch, cluster)
    intermediates.log_initial('step11')
    dirname_archive = os.path.dirname(flname) + '/archive'
    os.system('mkdir ' + dirname_archive)
    flsmiles_short = os.path.basename(flname)
    os.system('mv ' + flname + ' ' + dirname_archive + '/' + flsmiles_short)
    os.system("ls > /dev/null")
sys.exit()