#! /usr/bin/python
import sys
import glob
import liwei

"""

The last automated script in the electrophilic addition mechanism (Pd(OAc)+ added as an electrophile)

It works with all the files of step1 and step2, once all the cacluations are successfully completed, and generates the
text output of relative energies for all computed intermediates and the overall molecule reactivity rank

The script operates in a given working directory and requires cluster to be configured beforehand

"""

cur_dir = sys.argv[1]

Intermediates_calculated = []
Results = [] 

logfiles = [cur_dir + '/step1.log', cur_dir + '/step2.log']

scratch = '/scratch/lc640'
cluster = 'darwin_liwei'

for flsdf in glob.glob(cur_dir + "/*.sdf"):
    step1_log_upload = liwei.Loader(logfiles[0])
    step1_data = step1_log_upload.log_data_loader()
    step2_log_upload = liwei.Loader(logfiles[1])
    step2_data = step2_log_upload.log_data_loader()
    if step1_data.finished and step2_data.finished:
        molecule_load = liwei.Loader(flsdf)
        molecule = molecule_load.molecule()
        logfile1 = liwei.Output(logfiles[0])
        energy_molecule = logfile1.minimal_energy_molecule_list(0, scratch, cluster)
        logfile2 = liwei.Output(logfiles[1])
        energy_intermediate = logfile2.minimal_energy_molecule_list(1, scratch, cluster)
        difference = energy_intermediate - energy_molecule
        results = logfile2.intermediates_result_builder(molecule, 1, scratch, cluster)
        results.results_print(cur_dir, difference)
sys.exit()
        




    












