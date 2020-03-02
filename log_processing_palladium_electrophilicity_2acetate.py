#! /usr/bin/python
import sys
import os
import liwei

"""

The third automated script in the electrophilic addition mechanism (Pd(OAc)2 added as an electrophile)

It works with all the files of step2 (initial geometry optimisation of Pd-ipso intermediates) and updates log txt
file until all the calculations are complete

The script operates in a given working directory and requires cluster to be configured beforehand

"""

cur_dir = sys.argv[1]

logfiles = [cur_dir + '/step1.log', cur_dir + '/step2.log']

scratch = '/scratch/lc640'
cluster = 'darwin_liwei'

if os.path.isfile(logfiles[1]):
    log_data_load = liwei.Loader(logfiles[0])
    log_data = log_data_load.log_data_loader()
    log_data_updated = log_data.log_processing(0, scratch, cluster)
    log_data_updated.log_data_export(logfiles[0])
    log_data_load2 = liwei.Loader(logfiles[1])
    log_data2 = log_data_load2.log_data_loader()
    log_data_updated2 = log_data2.log_processing(0, scratch, cluster)
    log_data_updated2.log_data_export(logfiles[1])
sys.exit()
