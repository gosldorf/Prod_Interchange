# Prod_Interchange Repository

## Contains two code bases: interchange_simulation.py and antechamber_simulation.py

## Prod = Production ready

## Interchange = Openff toolkit parameterization method

## Antechamber = Antechamber+gaff2 parameterization api

### Contained within this repository is code, examples, and sbatch scripts needed to run Openmm using Openff-2.0 using the Interchange method to apply the desired forcefield parameters. 

**The contents of this repository are as follows:**

1. Code

Within this subdirectory is antechmaber_simulation.py which is the principle code base

And also, a version with openmm with Interchange_Simulation.py

Both codebases are different in that they use different methodologies to parameterize and create simulations systems. Both codebases include the ability to run openmm simulations in the codebase directly

2. Environment

Within this subdirectory are files relating to the defining of a conda environment suitable for running simulations using the code defined in this repository

3. Job_Submission

Within this subdirectory are sbatch script examples for running simulations on a slurm computational server

4. Examples

Within this subdirectory are a number of subdirectories within each are examples for running a variety of use-cases for this codebase

5. Sys_Prep

Within this subdirectory will be an example jupyter notebook within which we demonstrate preparing our pdbFile and sdfFile from the pdb directly downloaded from the rcsb database

### TODO

- Add to codebase the ability to run Amber simulations, have the code output Amber input files and then call pmemd to execute the calculations making it an all-in-one codebase

- Improve optimization

- Improve error message logging
