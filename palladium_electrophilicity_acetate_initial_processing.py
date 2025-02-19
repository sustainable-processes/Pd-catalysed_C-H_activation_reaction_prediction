#! /usr/bin/python
import sys
import os
import liwei

"""

The second automated script in the electrophilic addition mechanism (Pd(OAc)+ added as an electrophile)

It works with all the files of step11 (initial geometry optimisation of Pd-ipso intermediates) and updates log txt
file until all the calculations are complete and step2 calculations are submitted (final geometry optimisation
of Pd-ipso intermediates)

The script operates in a given working directory and requires cluster to be configured beforehand

"""

cur_dir = sys.argv[1]

scratch = '/scratch/lc640'
cluster = 'darwin_liwei'

logfile = cur_dir + "/step11.log"
log_data_load = liwei.Loader(logfile)
log_data = log_data_load.log_data_loader()
log_data = log_data.initial_processing_darwin()
if log_data.step_finished():
    log_data.log_data_export(logfile)
    log_data.initial_analysis(1, scratch, cluster)
    dirname_archive = os.path.dirname(logfile) + '/archive'
    logfile_short = os.path.basename(logfile)
    os.system('mv ' + logfile + ' ' + dirname_archive + '/' + logfile_short)
sys.exit()




