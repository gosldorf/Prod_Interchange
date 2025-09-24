# Code: Antechamber_Simulation.py

This codebase contains the methods defined towards the end goal of simulating in openmm a number of different system use-cases. These use-cases are specifically from the context of protein and CADD simulations. For additional use-cases the end-user will need to develop their own code, hack this codebase, or add to this codebase. Specifically using antechamber/gaff2 for ligand parameters, leap to construct systems, and Openmm for simulations currently.

# Code: Interchange_Simulation.py

This codebase contains the methods defined towards the end goal of simulating in openmm a number of different system use-cases. These use-cases are specifically from the context of protein and CADD simulations. For additional use-cases the end-user will need to develop their own code, hack this codebase, or add to this codebase.

## Interchange_Simulation.py and Antechamber_Simulation.py are divided roughly into 4 code blocks

1. Import block

Needed python libraries imported here

2. Platform block

Openmm performs optimally when the system platform is explicitly defined. In this block we define the desired platform to be CUDA. If the end-user desires other platform usage, edit this section or otherwise comment it out so that openmm can choose its own defaults...this may not be optimal.

3. Global Functions

In this block the global log is initialized as well as three global methods. These methods, insert_molecule_and_remove_clashes, add_posres, and add_ligposres perform valuable tasks that are outside the scope of the simulation protocol directly. The first, insert_molecule_and_remove_clashes, is used to embed the ligand molecule into the solvated protein system and to remove any overlapping atoms. **Note:** if the ligand is too close to the protein this will delete the protein!. The second, add_posres, is used to apply initial restraints as desired to the P-L or Apo system. The third, add_ligposres, does similar to the second, but instead to a ligand only simulation context.

4. Class simSystem

This is the final and principle block of code in this codebase. This class contains the stepwise methods and procedures which define our simulation protocol. More detail below


## simSystem

Our simulation protocol covers a number of use cases. Firstly is the definition of the system itself, this can be defined as: **P-L:** Protein-Ligand, **Apo:** Protein only, and **Lig:** Ligand only. Secondly is the definition of the desired restraints to be applied to the system. The options for restraints are: **None:** no restraints, **AHR:** all heavy atoms restrained, **BBR:** backbone restrained, **3DS:** three distal sites restrained (requires end-user definition).

To initialize a simSystem object the user must provide a pdbFile of the system (protein for P-L+Apo or ligand for Lig), a string defining the systype (P-L, Apo, Lig), an sdfFile of the ligand (can be empty file for Apo), and a restraintType (None, AHR, BBR, 3DS)

### simSystem Methods for both codebases:

Once initialized we can begin to utilize the methods within simSystem to undergo our simulation protocol

- solvateSystem(padding=1.0, ionicStrength=0.15)

Method takes the pdbFile provided upon initialization and creates a box of solvent around with a size defined by the padding argument (in nM) and with counter ions (NaCl by default) defined by the ionicStrength argument (in M)

- minimizeSystem(temperature=300, frictCoeff=1.0, stepSize=0.002, tolerance=50.0)

Method to minimized system in openmm using steepest descents method. Arguments define the parameters for running this openmm minimization under. Temperature is the simulation temperature (K), frictCoeff is the friction coefficient (1/ps), stepSize is the simulation step size (ps), tolerance is the desired convergance tolerance (kJ/mol/nM)

- heatSystem(Ti=100, Tf=300, numSims=5, mdsteps=250000,frictCoeff=1.0, stepSize=0.002,nc_reporter='heat.nc',state_reporter='heat.csv',saveFreq=1000)

Method to heat minimized system in NVT ensemble. Ti and Tf define the initial and final temperatures (K), numSims = number of simulation steps unto which this heating will be performed, mdsteps defines number of total simulation steps to be performed during heating. nc_reporter and state_reporter define output files openmm will write to during simulation. SaveFreq defines how frequently these reporters will be written to, 1000 corresponds to every 2 ps.

- releaseSystem(temperature=300, pressure=1, finalwt=2.5, numSims=5, mdsteps=250000, frictCoeff=1.0, stepSize=0.002, nc_reporter='release.nc', state_reporter='release.csv', saveFreq=1000)

Method to gradually release restraints in NPT ensemble. Pressure defines the simulation pressure (bar). Finalwt defines the desired restraintwt at the end of the releasing steps (kcal/mol/A^2). If no restraints this method should not be called.

- equilSystem(temperature=300, pressure=1, mdsteps=500000, frictCoeff=1, stepSize=0.002, nc_reporter='equilibrated.nc',state_reporter='equilibrated.csv', saveFreq=1000)

Method to equilibrate production ready simulation (minimized, heated, released as needed) in NPT ensemble

- prodSimulation(temperature=300, mdsteps=50000000, frictCoeff=1, stepSize=0.002, nc_reporter='prod.nc', state_reporter='prod.csv', saveFreq=500)

Production simulation method to be run following equilSystem. mdsteps 50,000,000 at 0.002 stepSize converts to 100 ns. SaveFreq defaults 500, which is every 1 ps.

### simSystem Methods for Interchange_Simulation.py:

- embedMolecule()

Method embeds the molecule in the provided sdfFile into the system after solvation. Not called in Apo contexts.

- createInterchange(listofForceFields=["openff-2.0.0.offxml", "ff14sb_off_impropers_0.0.3.offxml", "opc3-1.0.0.offxml"])

Method creates the interchange object and parameterizes created system using the forcefields provided in the listofForcefields argument. Defaults are amber14sb, openff-2.0, and opc3 respectively. To use alternate forcefields you will need to download or write the desired offxml in the offxml directory of the conda environment unto which you will call python from.

- addRestraints(restrainwt = 100, atomList=None)

Method to initialize restraints onto the system. Should only be called if the restraintType is not None. The default strength is 100 (kcal/mol/A^2). The atomList argument is intended for the restraintType 3DS context, otherwise the codebase would not know where to add restraints.

### simSystem Methods for Antechamber_Simulation.py:

Antechamber_Simulation has a larger number of methods for specifically creating different parameterized simulation systems. Each of these will output a set of topology files in prmtop and coordinate files in rst7. The methods differ in a few ways. First is how many ligands are handled, 0, 1, or multiple. Second is solvation character, vaccum, solvent, or solvent with ions. Each of these system building methods can be called in sequence, producing the topology and coordinate files for each of the contexts if desired, to quickly generate all the P-L systems desired.

- antechamberLigand(ligcharge=0)

Parameterize ligand with gaff2 charges. ligcharge defines the partial charge of the ligand needed to determine atomic charges

- antechamberMultipleLigand(ligcharge=[0], additionalLigands=[])

Parameterize multiple ligands with gaff2 charges. ligcharge defines the partial charge of the ligand needed to determine atomic charges

- createSolvatedLigand(padding=20)

Create a prmtop and rst7 of the provided ligand molecule in water, using the model defined at class intialization. Padding defines the buffer legnths of of the box.

- createSolvatedLigandMultiple(additionalLigands=[], padding=20)

Create a prmtop and rst7 of the provided ligand molecules in water, using the model defined at class intialization. Padding defines the buffer legnths of of the box. additionalLigands is a list of the multiple ligands to be handled

- createSolvatedProtein(padding=10)

Create a prmtop and rst7 of the provided protein in water, using the model defined at class intialization. Padding defines the buffer legnths of of the box.

- createIonsProtein(padding=10)

Create a prmtop and rst7 of the provided protein in water and ions, using the model defined at class intialization. Padding defines the buffer legnths of of the box.

- createSolvatedComplex(padding=10)

Create a prmtop and rst7 of the provided protein and ligand together in water, using the model defined at class intialization. Padding defines the buffer legnths of of the box.

- createSolvatedComplexMultiple(additionalLigands=[], padding=10)

Create a prmtop and rst7 of the provided protein and multiple ligands in water, using the model defined at class intialization. Padding defines the buffer legnths of of the box. additionalLigands is a list of the multiple ligands to be handled

- createVacuumComplex(padding=10)

Create a prmtop and rst7 of the provided protein and ligand in vacuum, using the model defined at class intialization. Padding defines the buffer legnths of of the box.

- createVacuumComplexMultiple(additionalLigands=[], padding=10)

Create a prmtop and rst7 of the provided protein and multiple ligands in vacuum, using the model defined at class intialization. Padding defines the buffer legnths of of the box. additionalLigands is a list of the multiple ligands to be handled

- createIonsComplex(padding=10)

Create a prmtop and rst7 of the provided protein and ligand in water and ions, using the model defined at class intialization. Padding defines the buffer legnths of of the box.

- createIonsComplexMultiple(additionalLigands=[], padding=10)

Create a prmtop and rst7 of the provided protein and ligand in water and ions, using the model defined at class intialization. Padding defines the buffer legnths of of the box. additionalLigands is a list of the multiple ligands to be handled
