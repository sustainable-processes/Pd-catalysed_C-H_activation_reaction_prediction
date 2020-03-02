# Pd-catalysed_C-H_activation_reaction_prediction
 This repository is associated to the paper "In silico rationalisation of selectivity and reactivity in Pd-catalysed C-H activation reactions"

### Getting started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

#### Prerequisites
NWChem software (http://www.nwchem-sw.org/index.php/Main_Page) need to be installed for DFT calculation

### Running the tests

The sturcture of starting molecules can be drawn in MarvinSketch (64bit) software and saved as a xyz file. And then, with OpenBabel 2.3.2, an open source chemistry toolbox, the xyz format file will be converted into different file formats for further calculation: smiles format, mol2 format, sdf format as well as conf format which contains conformations of the starting structure. Depending on different mechanisms, the starting stuctures will go through different paths within this program. The workflow is illustrated in the scheme below.

![image](https://user-images.githubusercontent.com/18735742/75668808-cb4b7000-5c71-11ea-82c0-9983e2dd1978.png)

### Deployment
The script operates in a given working directory and requires cluster to be configured beforehand.