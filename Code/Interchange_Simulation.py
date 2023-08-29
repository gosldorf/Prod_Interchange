########################Import Block########################
#general imports
import numpy as np
from io import StringIO
from typing import Iterable
from copy import deepcopy
import argparse
from pathlib import Path
#comp chem imports
import mdtraj
from pdbfixer import PDBFixer
import parmed as pmd
#openmm imports
import openmm
from openff.units import Quantity, unit
from openmm import unit as openmm_unit
#openff toolkit imports
from openff.toolkit import ForceField, Molecule, Topology
########################Import Block########################

########################Platform Block########################
##This section used by openmm to force accurate usage of CUDA
# without you may not receive optimal gpu md speeds
platform = openmm.Platform.getPlatformByName('CUDA')
properties={}
properties["DeviceIndex"]="0"
properties["Precision"]="mixed"
########################Platform Block########################

########################Global Functions########################
global_log = 'log_global.txt' #initialize logfile
glog = open(global_log, 'w')

## method stolen from openff toolkit example
def insert_molecule_and_remove_clashes(
    topology: Topology,
    insert: Molecule,
    radius: Quantity = 1.5 * unit.angstrom, #defines clash removal radius
    keep: Iterable[Molecule] = [],
) -> Topology:
    """
    Add a molecule to a copy of the topology, removing any clashing molecules.

    The molecule will be added to the end of the topology. A new topology is
    returned; the input topology will not be altered. All molecules that
    clash will be removed, and each removed molecule will be printed to stdout.
    Users are responsible for ensuring that no important molecules have been
    removed; the clash radius may be modified accordingly.

    Parameters
    ==========
    top
        The topology to insert a molecule into
    insert
        The molecule to insert
    radius
        Any atom within this distance of any atom in the insert is considered
        clashing.
    keep
        Keep copies of these molecules, even if they're clashing
    """
    # We'll collect the molecules for the output topology into a list
    new_top_mols = []
    # A molecule's positions in a topology are stored as its zeroth conformer
    insert_coordinates = insert.conformers[0][:, None, :]
    for molecule in topology.molecules:
        if any(keep_mol.is_isomorphic_with(molecule) for keep_mol in keep):
            new_top_mols.append(molecule)
            continue
        molecule_coordinates = molecule.conformers[0][None, :, :]
        diff_matrix = molecule_coordinates - insert_coordinates

        # np.linalg.norm doesn't work on Pint quantities 😢
        working_unit = unit.nanometer
        distance_matrix = (
            np.linalg.norm(diff_matrix.m_as(working_unit), axis=-1) * working_unit
        )

        if distance_matrix.min() > radius:
            # This molecule is not clashing, so add it to the topology
            new_top_mols.append(molecule)
        else:
            glog.write(f"Removed {molecule.to_smiles()} molecule")

    # Insert the ligand at the end
    new_top_mols.append(insert)

    # This pattern of assembling a topology from a list of molecules
    # ends up being much more efficient than adding each molecule
    # to a new topology one at a time
    new_top = Topology.from_molecules(new_top_mols)

    # Don't forget the box vectors!
    new_top.box_vectors = topology.box_vectors
    return new_top

# Function to add backbone position restraints
def add_posres(system, positions, atoms, restraint_force, res_context, atom_list = None):
    '''
    modified functiont to apply position restraints to openmm system
     restraints will be periodic distance forces applied with customized strengths
     on targeted atoms
     
    system = openmm system
    positions = openmm positions, might need modeller positions
    atoms = openmm atoms, might need modeller atoms
    restraint_force = desired force strength in (kcal/mol/Å)^2
    res_context = option to define restraints definition desired
        None = no restraints
        'AHR' = all heavy atoms
        'BBR' = backbone heavy atoms (CA, C, N)
        '3DS' = 3 distal sites (CA atoms)
    atom_list = list of atoms to apply restraints to (for custom defintions)
        required for '3DS'
        This list needs to be a list of strings so ['1','4'] for the first and fourth atom,
            start the count from 1 modeller resets i think
    '''
    #define list of protein residues to ensure we do not restrain other residues here (water/ligand)
    protein_residues = ['ACE','NME','NMA','ALA',
                        'ARG','ASN','ASP','ASH',
                        'CYS','CYX','CYM','GLN',
                        'GLU','GLH','GLY','HIS',
                        'HID','HIE','HIP','ILE',
                        'LEU','LYS','MET','PHE',
                        'PRO','SER','THR','TRP',
                        'TYR','VAL']
    force = openmm.CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    force_amount = restraint_force * openmm_unit.kilocalories_per_mole/openmm_unit.angstroms**2
    force.addGlobalParameter("k", force_amount)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    for i, (atom_crd, atom) in enumerate(zip(positions, atoms)):
        if str(atom).rsplit(' ')[-1][1:4] not in protein_residues:
            continue #skip this atom it aint a protein atom
        if res_context == 'BBR':
            if atom.name in  ('CA', 'C', 'N'):
                force.addParticle(i, atom_crd.value_in_unit(openmm_unit.nanometers))
        elif res_context == 'AHR':
            if 'hydrogen' not in str(atom.element):
                force.addParticle(i, atom_crd.value_in_unit(openmm_unit.nanometers))
        elif res_context == '3DS':
            if atom_list == None:
                glog.write('Error, 3DS chosen but no list of atoms provided. Need list of atom numbers to restrain\n')
                break
            if atom.id in atom_list:
                force.addParticle(i, atom_crd.value_in_unit(openmm_unit.nanometers))
    posres_sys = deepcopy(system)
    posres_sys.addForce(force)
    return posres_sys

def add_ligposres(system, positions, atoms, restraint_force, res_context, atom_list = None):
    '''
    modified function to apply position restraints to openmm system
     restraints will be periodic distance forces applied with customized strengths
     on targeted atoms
     
    system = openmm system
    positions = openmm positions, might need modeller positions
    atoms = openmm atoms, might need modeller atoms
    restraint_force = desired force strength in (kcal/mol/Å)^2
    res_context = option to define restraints definition desired
        None = no restraints
        'AHR' = all heavy atoms
        'BBR' = backbone heavy atoms (CA, C, N)
        '3DS' = 3 distal sites (CA atoms)
    atom_list = list of atoms to apply restraints to (for custom defintions)
        required for '3DS'
        This list needs to be a list of strings so ['1','4'] for the first and fourth atom, start the count from 1 modeller resets i think
    '''
    #define list of protein residues to ensure we do not restrain other residues here (water/ligand)
    water_residues = ['WAT','HOH','T3P','TP3','SPC']
    force = openmm.CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    force_amount = restraint_force * openmm_unit.kilocalories_per_mole/openmm_unit.angstroms**2
    force.addGlobalParameter("k", force_amount)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    for i, (atom_crd, atom) in enumerate(zip(positions, atoms)):
        if str(atom).rsplit(' ')[-1][1:4] in water_residues: continue
        elif res_context == 'AHR':
            if 'hydrogen' not in str(atom.element):
                force.addParticle(i, atom_crd.value_in_unit(openmm_unit.nanometers))
        elif res_context == '3DS':
            if atom_list == None:
                glog.write('Error, 3DS chosen but no list of atoms provided. Need list of atom numbers to restrain\n')
                break
            if atom.id in atom_list:
                force.addParticle(i, atom_crd.value_in_unit(openmm_unit.nanometers))
    posres_sys = deepcopy(system)
    posres_sys.addForce(force)
    return posres_sys

########################Global Functions########################


class simSystem():
    '''
    This class is the super structure unto which all of our protocol methods will go. 
    To use this class initialize a simSystem object with the following arguments:
    
    pdbFile: String containing file location of the desired system pdb. For P-L or Apo cases this will be a protein pdb
                for ligand only simulations this will be a pdb file of the ligand molecule.
             This provided pdbfile should be prepared and ready for use. See preparation examples for reference.
    sysType: This is a use case string that denotes the protocol that this object will be used to enact.
                See list of test cases in header to see list of use cases
             The accepted use cases are as follows:
                 "P-L": protein-ligand system (default)
                 "Apo": Apo protein (only protein, no sdf file required)
                 "Lig": Just ligand system
    sdfFile: String containing file location of the desired ligand molecule sdf. 
                Sdf must be in the correct location and with correct bond orders
                Default is None, but this will only be None for Apo systems. Required otherwise.
    
    restraintType: string that defines the restraints to be used on this system in these simulations
        Can be None for none, 'AHR' for all heavy atoms, 'BBR' for backbone atoms, or '3DS' for 3 distal sites
    restart = False, set to True to restart from a previous sim system pdb (written by interchange or openmm) In this case, the pdbFile should be the simsystem pdb not the protein.
    '''
    pdbFile = None
    sysType = "P-L"
    sdfFile = None
    restraintType = None
    ligand = None #storage for Molecule object
    solvatedTop = None #storage for after we build solvation system
    omm_top = None #storage for openmm topology object created after interchange parameterization
    omm_system = None #storage for openmm system object created after interchange parameterization
    minimizedState = None #storage for minimized system
    heatedState = None #storage for heated system
    releasedState = None #storage for 'released' system, ie loosening of restraints
    equilState = None #storage for equilibrated system
    lastState = None #storage for post production sim final frame for reference
    heatmdsteps = 0 #storage for number of simulation steps performed in heating
    releasemdsteps = 0 #storage for number of simulation steps performed in releasing restraints
    equilmdsteps = 0 #storage for number of simulation steps performed in equilibration
    prodmdsteps = 0 #storage for number of simulation steps performed in production
    totalmdsteps = 0 #storage for total mdsteps
    logfile = 'log_interchange_openmm.txt'
    log = open(logfile, 'w')
    def __init__(self,pdbFile,sysType="P-L",sdfFile=None,
                 restraintType=None,restart=False):
        '''
            init method, sets values fed by user in arguments into class variables, outputs error if necessary
        '''
        self.pdbFile = pdbFile
        self.sysType = sysType
        self.restraintType = restraintType
        if self.sysType not in ['P-L', 'Apo', 'Lig']:
            self.log.write("Error: Provided system type is not in the list of available options.\n")
            self.log.write("Please use one of: 'P-L', 'Apo', 'Lig'\n")
            return 1
        if self.restraintType is not None and self.restraintType not in ['AHR', 'BBR', '3DS']:
            self.log.write("Error: provided restraintType is not supported\n")
            self.log.write("Use either None (not string) or ['AHR', 'BBR', '3DS']\n")
            return 1
        self.sdfFile = sdfFile
        if self.sysType in ['P-L','Lig'] and self.sdfFile == None:
            self.log.write("Error: Selected system type contains a ligand, but no ligand sdf file provided\n")
            self.log.write("Please define ligand sdf file\n")
            return 1
        ##check that provided files exist and are reachable
        pathcheck1 = Path(pdbFile)
        if not pathcheck1.is_file():
            self.log.write("Error: Unable to read provided pdbfile\n")
            self.log.write("Check that path is correct\n")
            self.log.write(f"{pathcheck1}\n")
            return 1
        if self.sdfFile is not None: #dont need to check if we aren't using one
            pathcheck2 = Path(sdfFile)
            if not pathcheck2.is_file():
                self.log.write("Error: Unable to read provided sdffile\n")
                self.log.write("Check that path is correct\n")
                self.log.write(f"{pathcheck2}\n")
                return 1
            self.ligand = Molecule.from_file(sdfFile) #initialize Molecule object
        ##made it this far, then everything initialized properly. Output status to command line
        if restart:
            if self.sysType=='P-L':
                listofForcefields=["openff-2.0.0.offxml",
                               "ff14sb_off_impropers_0.0.3.offxml",
                               "opc3-1.0.0.offxml"
                ]
                sage = ForceField(listofForcefields[0],
                              listofForcefields[1],
                              listofForcefields[2])
                self.ligand = Molecule.from_file(self.sdfFile)
                self.solvatedTop = Topology.from_pdb(self.pdbFile,
                                                 unique_molecules=[self.ligand])
            elif self.sysType=='Apo':
                listofForcefields=["ff14sb_off_impropers_0.0.3.offxml",
                               "opc3-1.0.0.offxml"
                ]
                sage = ForceField(listofForcefields[0],
                              listofForcefields[1])
                self.solvatedTop = Topology.from_pdb(self.pdbFile)
            else:
                listofForcefields=["openff-2.0.0.offxml",
                               "opc3-1.0.0.offxml"
                ]
                sage = ForceField(listofForcefields[0],
                              listofForcefields[1])
                self.ligand = Molecule.from_file(self.sdfFile)
                self.solvatedTop = Topology.from_pdb(self.pdbFile,
                                                 unique_molecules=[self.ligand])
            with open("topology.pdb", "w") as f:
                print(self.solvatedTop.to_file(file=f))
            interchange = sage.create_interchange(self.solvatedTop)
            self.omm_system = interchange.to_openmm()
            self.omm_top = interchange.to_openmm_topology()
            self.log.write("Restart system reinitilized successfully!\n")
            self.log.write("Created with the following inputs\n")
            self.log.write(f"pdb: {self.pdbFile}\n")
            self.log.write(f"sdf: {self.sdfFile}\n")
            self.log.write(f"sysType: {self.sysType}\n")
            self.log.write(f"restraintType: {self.restraintType}\n")
            self.log.write("Ready to continue protocol!\n")
        else:
            self.log.write("Simulation system object created successfully!\n")
            self.log.write("Created with the following inputs\n")
            self.log.write(f"pdb: {self.pdbFile}\n")
            self.log.write(f"sdf: {self.sdfFile}\n")
            self.log.write(f"sysType: {self.sysType}\n")
            self.log.write(f"restraintType: {self.restraintType}\n")
            self.log.write("Ready to continue protocol!\n")
        return 
        
    
    def solvateSystem(self, padding=1.0, ionicStrength=0.15):
        '''
        Method that calls pdbfixer so as to solvate the box around the provided pdbFile
        Creating a solvated simulation system requires a few parameters
            in this case pdbfixer is creating a rectangular box by default
        
        padding: The distance from the edge of the atoms in the pdbFile to edge of the box (nM)
        
        ionicStrength: desired ionic strength (M), this will also neutralize charge. If 0, will not add ions.
            Highly recommend adding ions to neutralize charge in systems containig proteins. 
                Industry standard is for around 0.15 Molar, hence default.
            Not recommended in Just ligand systems.
        '''
        fixer = PDBFixer(self.pdbFile)
        if ionicStrength == 0: #user indicated no ions, will not add
            fixer.addSolvent(
                padding=padding * openmm_unit.nanometer
            )
        else: #user indicated ions, will add
            fixer.addSolvent(
                padding=padding * openmm_unit.nanometer, ionicStrength = ionicStrength * openmm_unit.molar
            )
        with open("solvatedSystem.pdb", "w") as f: #save intermediate pdb file, user should check this
            openmm.app.PDBFile.writeFile(fixer.topology, fixer.positions, f)
        if self.sysType == "Lig":
            self.ligand = Molecule.from_file(self.sdfFile)
            self.solvatedTop = Topology.from_pdb("solvatedSystem.pdb",
                                                 unique_molecules=[self.ligand])
        else:
            self.solvatedTop = Topology.from_pdb("solvatedSystem.pdb")
        with open("topology.pdb", "w") as f: #placeholder for apo/ligand situations, should be replaced by embedMolecule in P-L systems
            print(self.solvatedTop.to_file(file=f))
        return 0
    
    def embedMolecule(self):
        '''
        Method that calls 'insert_molecule_and_remove_clashes'
        Will not be needed unless running a P-L system.
        Method assumes user ran solvate system first
        '''
        pathcheck3 = Path('solvatedSystem.pdb') #check previous step intermediate is reachable, error if not
        if not pathcheck3.is_file():
            self.log.write("Error! User initiated embedMolecule method without producing solvatedSystem first\n")
            self.log.write("Method requires a solvated system to embed into\n")
            self.log.write("Please run solvateSystem or ensure 'solvatedSystem.pdb' is reachable\n")
            return 1
        if self.sysType == "P-L":
            self.solvatedTop = insert_molecule_and_remove_clashes(self.solvatedTop, self.ligand)
        with open("topology.json", "w") as f:
            print(self.solvatedTop.to_json(), file=f)
        with open("topology.pdb", "w") as f:
            print(self.solvatedTop.to_file(file=f))
        return 0
    
    def createInterchange(self, listofForcefields=[
        "openff-2.0.0.offxml",
        "ff14sb_off_impropers_0.0.3.offxml",
        "opc3-1.0.0.offxml"
    ]):
        '''
        Method that takes the embedded or solvated system depending on sysType
            and creates an interchange object with it
        listofForcefields: list of openff offxml files to use when building interchange object
            by default using openff-2.0, ff14sb, and opc3 models
            user can provide own list
            
        opc3 offxml file needs to be downloaded from https://github.com/openforcefield/openff-forcefields/tree/main/openforcefields/offxml
            and put into the /envs/openff/lib/python3.10/site-packages/openforcefields/offxml directory
        '''
        #there will always be at least 2 forcefields in use (water+ligand, water+protein)
        if len(listofForcefields) < 2:
            self.log.write("Error: Not enough forcefields provided. Need at least 2\n")
            self.log.write("Please provide offxml for water and protein/ligand\n")
            return 1
        if len(listofForcefields) < 3 and self.sysType=='P-L':
            self.log.write("Error: Not enough forcefields provided. Need at least 3 for P-L system\n")
            self.log.write("Please provide offxml for water, protein, and ligand\n")
            return 1
        if len(listofForcefields) < 3:
            self.log.write(f"parameterizing system with {listofForcefields[0]}, {listofForcefields[1]}\n")
            sage = ForceField(listofForcefields[0], listofForcefields[1])
        else:
            self.log.write(f"parameterizing system with {listofForcefields[0]}, {listofForcefields[1]}, {listofForcefields[2]}\n")
            sage = ForceField(listofForcefields[0], listofForcefields[1], listofForcefields[2])
        interchange = sage.create_interchange(self.solvatedTop)
        self.omm_system = interchange.to_openmm()
        self.omm_top = interchange.to_openmm_topology()
        return 0
    
    def addRestraints(self, restraintwt = 100, atomList = None):
        '''
        Function for adding restraints to the system using the add_posres or add_ligposres functions
        Requires previous steps had been run to create 'topology.pdb' and omm_top, omm_system
        
        restraintContext: Choose the restraint context to use:
            AHR = all heavy atoms restrained,
            BBR = backbone restrained,
            3DS = 3 distal sites (requires atomList)
            
        restraintwt: weight of restraints to be added (kcal/mol/Å^2)
        
        atomList: list of atoms to restrain in 3DS (can be used to customize restraints, must be provided)
        '''
        pathcheck4 = Path('topology.pdb')
        if not pathcheck4.is_file():
            self.log.write("Error! User initiated addRestraints without first generating a topology.pdb\n")
            self.log.write("Method requires a reference topology input file\n")
            self.log.write("Please run solvateSystem or ensure 'topology.pdb' is reachable\n")
            return 1
        if self.omm_system == None or self.omm_top == None:
            self.log.write("Error: simSystem has not been fully initialized!\n")
            self.log.write("Cannot add restraints a system if we have not yet built one.\n")
            self.log.write("Please perform preceding steps, namely createInterchange before this method\n")
            return 1
        if self.restraintType == '3DS' and atomList == None:
            self.log.write("Error: Cannot use 3DS restraint context without a provided list of atoms to restrain\n")
            self.log.write("atomList needs to be provided, format is ['1','1273','678'] (list of numerical strings)\n")
            return 1
        pdb = openmm.app.PDBFile('topology.pdb')
        modeller = openmm.app.Modeller(pdb.topology, pdb.positions)
        if self.sysType == 'Lig': #use add_ligposres
            self.log.write(f'Adding restraints, {self.restraintType}, {restraintwt}\n')
            self.omm_system = add_ligposres(self.omm_system, modeller.positions,
                                            modeller.topology.atoms(), restraintwt,
                                            self.restraintType, atomList)
        else:
            self.log.write(f'Adding restraints, {self.restraintType}, {restraintwt}\n')
            self.omm_system = add_posres(self.omm_system, modeller.positions,
                                         modeller.topology.atoms(), restraintwt,
                                         self.restraintType, atomList)
        return 0
    
    def minimizeSystem(self, temperature=300, frictCoeff=1.0, stepSize=0.002, tolerance=50.0):
        '''
        Function for minimizing the system after openmm system has been made/parameterized
            Function will error if self.omm_system/top not defined
        
        temperature = desired temperature of the minimization, default 300 (K)
        frictCoeff = desired friction coeffecient, default 1 (1/ps)
        stepSize = desired simulation step size, default 0.002 (ps)
        tolerance = desired energetic tolerance for minimization, default 50 (kJ/mol/nM)
        '''
        if self.omm_system == None or self.omm_top == None:
            self.log.write("Error: simSystem has not been fully initialized!\n")
            self.log.write("Cannot minimize a system if we have not yet built one.\n")
            self.log.write("Please perform preceding steps, namely createInterchange before this method\n")
            return 1
        integrator = openmm.LangevinIntegrator(
            temperature * openmm_unit.kelvin,
            frictCoeff / openmm_unit.picosecond,
            stepSize * openmm_unit.picoseconds
        )
        simulation = openmm.app.Simulation(self.omm_top, self.omm_system, integrator)
        simulation.context.setPositions(self.solvatedTop.get_positions().to_openmm())
        nc_reporter = pmd.openmm.NetCDFReporter("min.nc", 10)
        simulation.reporters.append(nc_reporter)
        simulation.minimizeEnergy(
            tolerance = openmm_unit.Quantity(
                value=tolerance, unit=openmm_unit.kilojoule_per_mole / openmm_unit.nanometer
            )
        )
        self.minimizedState = simulation.context.getState(
            getPositions=True, getEnergy=True, getForces=True
        )
        positions = simulation.context.getState(getPositions=True).getPositions()
        openmm.app.PDBFile.writeFile(simulation.topology, positions, open('minimized.pdb', 'w'))
        self.log.write(
            f"Minimized to,{self.minimizedState.getPotentialEnergy()},with maximum force,{max(np.sqrt(v.x * v.x + v.y * v.y + v.z * v.z) for v in self.minimizedState.getForces())},{self.minimizedState.getForces().unit.get_symbol()}"
        )
        return 0
        
    def heatSystem(self, Ti = 100, Tf = 300, numSims=5, mdsteps=250000, frictCoeff=1.0, stepSize=0.002,
                   nc_reporter="heat.nc", state_reporter="heat.csv", saveFreq=1000):
        '''
        Method intended for heating the system from low temps (as in static xtal structures to production temp)
            Method assumes minimizeSystem was run already, will fail if self.minimizedState is not defined
            
        Ti: initial temperature, default 100 (K)
        Tf: final production simulation temperature, default 300 (K)
        numSims: number of heating steps to perform, this defines the different T we sample at effectively.
            This includes a short sim at Ti and Tf so the default of 5 means [Ti, Ti+C, Ti+2C, Ti+3C, Tf]
        mdsteps: total number of simulation steps to be performed in this function, default 250,000 -- 500 ps
        frictCoeff: friction coefficient, default 1.0 (1/ps)
        stepSize: simulation timestep, default 0.002 (ps)
        nc_reporter: defined file output for netcdf file, default 'heat.nc'
        state_reporter: defined file output for logfile, default 'heat.csv'
        saveFreq: how often should we write to the nc and state reporters. Default 1000 steps, or every 2 ps
        
        If user defines mdsteps/numSims manually it is the users responsibility to ensure that the mdsteps are divisible by numSims
        '''
        if self.omm_system == None or self.omm_top == None:
            self.log.write("Error: simSystem has not been fully initialized!\n")
            self.log.write("Cannot minimize a system if we have not yet built one.\n")
            self.log.write("Please perform preceding steps, namely createInterchange before this method\n")
            return 1
        if self.minimizedState == None:
            self.log.write("Error: simSystem has not been minimized yet!\n")
            self.log.write("Heating a system in this protocol presumes a minimized system was first created\n")
            self.log.write("Please use minimizeSystem before using this function\n")
            return 1
        integrator = openmm.LangevinIntegrator(
            Tf * openmm_unit.kelvin,
            frictCoeff / openmm_unit.picosecond,
            stepSize * openmm_unit.picoseconds
        )
        self.heatmdsteps = mdsteps+(mdsteps/numSims)
        simulation = openmm.app.Simulation(self.omm_top, self.omm_system,
                                           integrator,platform,properties)
        simulation.context.setState(self.minimizedState)
        simulation.context.setVelocitiesToTemperature(Ti * openmm_unit.kelvin)
        nc_reporter = pmd.openmm.NetCDFReporter(nc_reporter, saveFreq)
        simulation.reporters.append(openmm.app.StateDataReporter(state_reporter, saveFreq, step=True,
                                                                potentialEnergy=True, kineticEnergy=True,totalEnergy=True,
                                                                temperature=True,volume=True,density=True,progress=True,
                                                                separator=',',totalSteps=int(self.heatmdsteps)))
        simulation.reporters.append(nc_reporter)
        T = Ti
        jumps = int((Tf-Ti)/numSims)
        for i in range(numSims):
            temperature = (T + i*jumps) * openmm_unit.kelvin
            integrator.setTemperature(temperature)
            simulation.step(int(mdsteps/numSims))
        temperature = Tf
        integrator.setTemperature(temperature)
        simulation.step(int(mdsteps/numSims))
        self.heatedState = simulation.context.getState(
            getPositions=True, getEnergy=True, getForces=True
        )
        positions = simulation.context.getState(getPositions=True).getPositions()
        openmm.app.PDBFile.writeFile(simulation.topology, positions, open('heated.pdb', 'w'))
        return 0
    
    def releaseSystem(self, temperature=300, pressure=1, finalwt = 2.5,
                      numSims = 5, mdsteps = 250000, frictCoeff = 1.0, stepSize = 0.002,
                      nc_reporter='release.nc', state_reporter='release.csv', saveFreq=1000):
        '''
        Method for gradually releasing restraints in NPT equilibration simulations following heating steps
        
        temperature: simulation temperature, default 300 (K)
        pressure: simulation pressure, default 1 (bar)
        finalwt: final desired restraint wt, default 2.5 (kcal/mol/Å^2)
        numSims: number of steps to perform the releasing of restraints, default 5
        mdsteps: total number of simulation steps to be performed by this function, default 250000 (500 ps)
        frictCoeff: friction coefficient, default 1 (1/ps)
        stepSize: simulation timestep, default 0.002 (ps)
        nc_reporter: defined trajectory output file, default 'release.nc'
        state_reporter: defined logfile output, default 'release.csv'
        saveFreq: Number of steps between each log/trajectory write, default 1000 (every 2 ps)
        '''
        if self.restraintType == None:
            self.log.write("Error: Attempting to run a restraint releasing simulation on a system that is unrestrained\n")
            return 1
        integrator = openmm.LangevinIntegrator(
            temperature * openmm_unit.kelvin,
            frictCoeff / openmm_unit.picosecond,
            stepSize * openmm_unit.picoseconds
        )
        self.releasemdsteps = mdsteps + int(mdsteps/numSims) #perform an extra at final restraints
        self.omm_system.addForce(openmm.MonteCarloBarostat(pressure * openmm_unit.bar, temperature*openmm_unit.kelvin))
        simulation = openmm.app.Simulation(self.omm_top, self.omm_system,
                                           integrator,platform,properties)
        simulation.context.setState(self.heatedState)
        nc_reporter = pmd.openmm.NetCDFReporter(nc_reporter, saveFreq)
        simulation.reporters.append(openmm.app.StateDataReporter(state_reporter, saveFreq, step=True,
                                                                potentialEnergy=True, kineticEnergy=True,totalEnergy=True,
                                                                temperature=True,volume=True,density=True,progress=True,
                                                                separator=',',totalSteps=int(self.heatmdsteps + self.releasemdsteps)))
        simulation.reporters.append(nc_reporter)
        jumps = int((100-finalwt)/numSims)
        for i in range(numSims):
            simulation.context.setParameter('k', (float(95-(i*10))*openmm_unit.kilocalories_per_mole/openmm_unit.angstroms**2))
            simulation.step(int(mdsteps/numSims))
        simulation.context.setParameter('k', float(finalwt)*openmm_unit.kilocalories_per_mole/openmm_unit.angstroms**2)
        simulation.step(int(mdsteps/numSims))
        self.releasedState = simulation.context.getState(
            getPositions=True, getEnergy=True, getForces=True
        )
        positions = simulation.context.getState(getPositions=True).getPositions()
        openmm.app.PDBFile.writeFile(simulation.topology, positions, open('released.pdb', 'w'))
        return 0
    
    def equilSystem(self, temperature=300, pressure=1, mdsteps=500000, frictCoeff=1, stepSize=0.002,
                    nc_reporter='equilibrated.nc',state_reporter='equilibrated.csv', saveFreq=1000):
        '''
        Method for running equilibration simulation in NPT on the system
        
        temperature: simulation temperature, default 300 (K)
        pressure: simulation pressure, default 1 (bar)
        mdsteps: number of simulation steps, default 500000 (1 ns)
        frictCoeff: integrator friction coefficient, default 1 (1/ps)
        stepSize: simulation step size, default 0.002 (ps)
        nc_reporter: trajectory output file, default 'equilibrated.nc'
        state_reporter: logfile output file, default 'equilibrated.csv'
        saveFreq: frequency to save to logfiles/trajectory, default 1000 (every 2 ps)
        '''
        integrator = openmm.LangevinIntegrator(
            temperature * openmm_unit.kelvin,
            frictCoeff / openmm_unit.picosecond,
            stepSize * openmm_unit.picoseconds
        )
        self.equilmdsteps = mdsteps
        if self.restraintType == None:
            self.omm_system.addForce(openmm.MonteCarloBarostat(pressure*openmm_unit.bar, temperature*openmm_unit.kelvin))
        simulation = openmm.app.Simulation(self.omm_top, self.omm_system,
                                           integrator,platform,properties)
        if self.restraintType == None:
            simulation.context.setState(self.heatedState)
        else:
            simulation.context.setState(self.releasedState)
        nc_reporter = pmd.openmm.NetCDFReporter(nc_reporter, saveFreq)
        simulation.reporters.append(openmm.app.StateDataReporter(state_reporter, saveFreq, step=True,
                                                                potentialEnergy=True,kineticEnergy=True,totalEnergy=True,
                                                                temperature=True,volume=True,density=True,progress=True,
                                                                separator=',',totalSteps=int(self.heatmdsteps + self.releasemdsteps + self.equilmdsteps)))
        simulation.reporters.append(nc_reporter)
        simulation.step(self.equilmdsteps)
        self.equilState = simulation.context.getState(
            getPositions=True, getEnergy=True, getForces=True
        )
        positions = simulation.context.getState(getPositions=True).getPositions()
        openmm.app.PDBFile.writeFile(simulation.topology, positions, open('equilibrated.pdb', 'w'))
        
    def prodSimulation(self, temperature=300, mdsteps=50000000, frictCoeff=1, stepSize=0.002,
                      nc_reporter='prod.nc',state_reporter='prod.csv',saveFreq=500):
        '''
        Method for running production simulations in NVT systems
        
        temperature: simulation temperature, default 300 (K)
        mdsteps: number of md steps to be performed, default 50,000,000 (total 100 ns)
        frictCoeff: integrator frictional coefficient, default 1 (1/ps)
        stepSize: simulation step size, default 0.002 (ps)
        nc_reporter: trajectory file output, default 'prod.nc'
        state_reporter: trajectory log output, default 'prod.csv'
        saveFreq: frequency to write to the log/trajectory, default 500 (every 1 ps)
        '''
        integrator = openmm.LangevinIntegrator(
            temperature * openmm_unit.kelvin,
            frictCoeff / openmm_unit.picosecond,
            stepSize * openmm_unit.picoseconds
        )
        self.prodmdsteps = mdsteps
        simulation = openmm.app.Simulation(self.omm_top, self.omm_system,
                                           integrator,platform,properties)
        simulation.context.setState(self.equilState)
        if self.restraintType == None:
            simulation.system.getForce(4).setFrequency(0)
        else:
            simulation.system.getForce(5).setFrequency(0)
        nc_reporter = pmd.openmm.NetCDFReporter(nc_reporter, saveFreq)
        simulation.reporters.append(openmm.app.StateDataReporter(state_reporter, saveFreq, step=True,
                                   potentialEnergy=True,kineticEnergy=True,totalEnergy=True,
                                   temperature=True,volume=True,density=True,
                                   progress=True,separator=',',
                                   totalSteps=self.heatmdsteps+self.releasemdsteps+self.equilmdsteps+self.prodmdsteps))
        simulation.reporters.append(nc_reporter)
        simulation.step(self.prodmdsteps)
        self.lastState = simulation.context.getState(
            getPositions=True, getEnergy=True, getForces=True
        )
        positions = simulation.context.getState(getPositions=True).getPositions()
        openmm.app.PDBFile.writeFile(simulation.topology, positions, open('last_frame.pdb', 'w'))
        self.log.close()
