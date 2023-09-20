########################Import Block########################
#general imports
import numpy as np
from io import StringIO
from typing import Iterable
from copy import deepcopy
from pathlib import Path
import os
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
from openff.interchange import Interchange
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


# Function to add backbone position restraints
def add_posres(system, positions, atoms, restraint_force, res_context, atom_list = None):
    '''
    modified function to apply position restraints to openmm system
     restraints will be periodic distance forces applied with customized strengths
     on targeted atoms
     
    system = openmm system
    positions = openmm positions, might need modeller positions
    atoms = openmm atoms, might need modeller atoms
    restraint_force = desired force strength in (kcal/mol/Å)^2
    res_context = option to define restraints definition desired
        'AHR' = all heavy atoms
        'BBR' = backbone heavy atoms (CA, C, N)
        '3DS' = 3 distal sites (CA atoms)
    atom_list = list of atoms to apply restraints to (for custom defintions)
        required for '3DS'
        This list needs to be a list of strings so ['1','4'] for the first and fourth atom,
            start the count from 1 for modeller 
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
        'AHR' = all heavy atoms
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
    ligFile = None
    restraintType = None
    ligFF = None
    proFF = None
    watFF = None
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
    def __init__(self,pdbFile,sysType="P-L",ligFile=None, sdfFile=None,
                 restraintType=None,restart=False, proFF = 'protein.ff19SB', ligFF = 'gaff2', watFF = 'water.opc', ions=0.15):
        '''
            init method, sets values fed by user in arguments into class variables, outputs error if necessary
        '''
        self.pdbFile = pdbFile
        self.sysType = sysType
        self.ligFile = ligFile
        self.sdfFile = sdfFile
        self.ligand = Molecule.from_file(self.sdfFile)
        self.proFF = proFF
        self.ligFF = ligFF
        self.watFF = watFF
        self.restraintType = restraintType
        self.ions = ions ##desired ion M, 0 for counter ions only, None for no ions whatsoever
        self.prmtop = None
        self.inpcrd = None
        if self.sysType not in ['P-L', 'Apo', 'Lig']:
            self.log.write("Error: Provided system type is not in the list of available options.\n")
            self.log.write("Please use one of: 'P-L', 'Apo', 'Lig'\n")
            return 1
        if self.restraintType is not None and self.restraintType not in ['AHR', 'BBR', '3DS']:
            self.log.write("Error: provided restraintType is not supported\n")
            self.log.write("Use either None (not string) or ['AHR', 'BBR', '3DS']\n")
            return 1
        if self.sysType in ['P-L','Lig'] and self.sdfFile == None:
            self.log.write("Error: Selected system type contains a ligand, but no ligand sdf file provided\n")
            self.log.write("Please define ligand sdf file\n")
            return 1
        elif self.sysType in ['P-L','Lig'] and self.ligFile == None:
            self.log.write("Error: Selected system type contains a ligand, but no ligand pdb file provided\n")
            self.log.write("Please define ligand pdb file\n")
            return 1
        else: #provided ligand and sdf
            pathcheck2 = Path(sdfFile)
            pathcheck3 = Path(ligFile)
            if not pathcheck2.is_file():
                self.log.write("Error: Unable to read provided sdffile\n")
                self.log.write("Check that path is correct\n")
                self.log.write(f"{pathcheck2}\n")
                return 1
            if not pathcheck3.is_file():
                self.log.write("Error: Unable to read provided ligfile\n")
                self.log.write("Check that path is correct\n")
                self.log.write(f"{pathcheck3}\n")
                return 1
        ##check that provided files exist and are reachable
        pathcheck1 = Path(pdbFile)
        if not pathcheck1.is_file():
            self.log.write("Error: Unable to read provided pdbfile\n")
            self.log.write("Check that path is correct\n")
            self.log.write(f"{pathcheck1}\n")
            return 1
        ##made it this far, then everything initialized properly. Output status to command line
        ##TODO: edit restart lines
        '''
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
            '''
        return 
        

    def antechamberLigand(self, ligcharge=0):
        '''
        Method parameterizes ligand using standard antechamber/sqm methodology outlined in Amber Tutorials
        user has already specified ligand input (ligFile)

        ligcharge: charge of ligand to be fed to antechamber -nc flag, default is 0. antechamber struggles with abs(>2) charge
        '''
        if self.sysType == 'Apo':
            self.log.write('Error: user selected to parameterize a ligand, but to run without a ligand\n')
            self.log.write('Would recommend syncing the run context options\n')
            return 1
        os.system(f'antechamber -i {self.ligFile} -fi pdb -o ligand.mol2 -fo mol2 -c bcc -nc {ligcharge} -rn LIG -at gaff2')
        sqmpath = Path('sqm.pdb')
        if not sqmpath.is_file():
            self.log.write('sqm did not successfully calculate charges for ligand\n')
            self.log.write('check logs and inputs\n')
            return 1
        os.system('rm -rf ANTE*') #remove temp files
        os.system('rm -rf ATOMTYPE.INF')
        os.system('parmchk2 -i ligand.mol2 -f mol2 -o LIG.frcmod -s gaff2 -a Y')
        #write our tleap input file from our python script
        with open('tleap1.in','w') as file:
            l1 = f"source leaprc.{self.ligFF}\n"
            l2 = "loadamberparams LIG.frcmod\n" 
            l3 = "ligand = loadmol2 ligand.mol2\n" 
            l4 = "saveoff ligand LIG.lib\n"
            l5 = "saveamberparm ligand ligand_vac.prmtop ligand_vac.rst7\n"
            l6 = "quit"
            file.writelines([l1, l2, l3, l4, l5, l6])
        os.system("tleap -f tleap1.in") #run tleap from our python script
        self.log.write('created ligand in vacuum prmtop successfully\n')
        return
        

    def createSolvatedLigand(self, padding=20): ##padding in Å
        '''
        Method to use parameters generated in antechamberLigand to generate a solvated prmtop of the ligand

        padding: extents of box padding to be fed to tleap solvateBox command, default here is 20 Å because ligands are small, 15 is good too.
        '''
        boxstring = self.watFF[6:].upper() + 'BOX'
        with open('tleap2.in','w') as file:
            l1 = f"source leaprc.{self.ligFF}\n"
            l2 = f"source leaprc.{self.watFF}\n"
            l3 = "loadamberparams LIG.frcmod\n" 
            l4 = "loadoff LIG.lib\n"
            l5 = "ligand = loadmol2 ligand.mol2\n"
            l6 = f"solvatebox ligand {boxstring} {padding}\n"
            l7 = "saveamberparm ligand ligand_wat.prmtop ligand_wat.rst7\n"
            l8 = "quit"
            file.writelines([l1, l2, l3, l4, l5, l6, l7, l8])
        os.system('tleap -f tleap2.in')
        self.log.write('created ligand in solvent prmtop successfully\n')
        return

    def createSolvatedProtein(self, padding=10): ##padding in Å
        '''
        Method to generate a standard protein in water prmtop with tleap using the provided protein file

        padding: extents of box padding to be fed to tleap solvateBox command, default here is 10 Å, wouldn't go too much larger
        '''
        boxstring = self.watFF[6:].upper() + 'BOX'
        os.system('rm -rf leap.log') #clean leap log so we can search info about these steps more easily afterwards
        with open('tleap3.in','w') as file:
            l1 = f"source leaprc.{self.proFF}\n"
            l2 = f"source leaprc.{self.watFF}\n"
            l3 = f"protein = loadpdb {self.pdbFile}\n" ##needed since we include xtal waters
            l4 = f"solvatebox protein {boxstring} {padding}\n"
            l5 = "saveamberparm protein protein_wat.prmtop protein_wat.rst7\n"
            l6 = "quit"
            file.writelines([l1, l2, l3, l4, l5, l6])
        os.system("tleap -f tleap3.in") #run tleap from our python script
        self.log.write('created protein in solvent prmtop successfully\n')
        return

    def createIonsProtein(self, padding=10, targetM = 0.15):
        '''
        Method to generate a standard protein in water with counter ions (added to desired molarity)

        padding: extents of box padding to be fed to tleap solvateBox command, default here is 10 Å, wouldn't go too much larger
        targetM: target ion molarity (M), default 0.15, but if set to 0 will only add counter ions to neutralize charge of system 
        '''
        boxstring = self.watFF[6:].upper() + 'BOX'
        ### first use leap.log from solvation of protein to determine system charge
        charge = 0
        with open('leap.log', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'The unperturbed charge of the unit' in line:
                    print(line)
                    splits = line.rsplit()
                    charge = int(float(splits[6][1:-1]))
                    print(int(float(splits[6][1:-1])))
        if charge == 0:
            context = 'uncharged'
        elif charge < 0:
            context = 'negative'
        elif charge > 0:
            context = 'positive'
        ### calculate number of ions to add to the system to achieve desired targetM
        with open('protein_wat.rst7', 'r') as f:
            last_line = f.readlines()[-1]
        raw = last_line.rsplit()
        protein_wat_boxcoord = []
        for i in range(3):
            protein_wat_boxcoord.append(float(raw[i]))
        print(f'coordinates of box: {protein_wat_boxcoord}')
        protein_wat_boxvol = protein_wat_boxcoord[0]*protein_wat_boxcoord[1]*protein_wat_boxcoord[2]
        protein_wat_boxvol_L = protein_wat_boxvol * float(1/(10**10)**3) * float((10**2)**3/1) * float(1/10**3)
        target_molarity = targetM #M
        target_molarity_mmol = target_molarity*1000
        target_ion_atoms__L = target_molarity_mmol * float(1/10**3) * float(6.022*10**23)
        target_num_ions = int(protein_wat_boxvol_L * target_ion_atoms__L)
        print(f'volume of box (Å^2): {protein_wat_boxvol}')
        print(f'volume of box (L): {protein_wat_boxvol_L}')
        print(f'target ionic concentration (mmol): {target_molarity_mmol}')
        print(f'target number of ions (atoms/L): {target_ion_atoms__L}')
        if context == 'uncharged':
            print(f'target number of ions (atoms): {target_num_ions} Na, {target_num_ions} Cl')
            numNa = target_num_ions
            numCl = target_num_ions
        if context == 'negative':
            print(f'target number of ions (atoms): {target_num_ions+abs(charge)} Na, {target_num_ions} Cl')
            numNa = target_num_ions+abs(charge)
            numCl = target_num_ions
        if context == 'positive':
            print(f'target number of ions (atoms): {target_num_ions} Na, {target_num_ions+abs(charge)} Cl')
            numNa = target_num_ions
            numCl = target_num_ions+abs(charge)
        if targetM==0.0:
            #user has specified to add no additional ions, only counters
            if context == 'uncharged':
                numNa = 0
                numCl = 0
            elif context == 'negative':
                numNa = 0+abs(charge)
                numCl = 0
            elif context == 'positive':
                numNa = 0
                numCl = 0+abs(charge)

        #now tleap
        with open('tleap4.in','w') as file:
            l1 = f"source leaprc.{self.proFF}\n"
            l2 = f"source leaprc.{self.watFF}\n"
            l3 = f"protein = loadpdb {self.pdbFile}\n" 
            l4 = f"solvatebox protein {boxstring} {padding}\n"
            l5 = f'addIonsRand protein Na+ {numNa} Cl- {numCl}\n'
            l6 = "saveamberparm protein protein_ion.prmtop protein_ion.rst7\n"
            l7 = "quit"
            file.writelines([l1, l2, l3, l4, l5, l6, l7])
            
        os.system("tleap -f tleap4.in")
        self.log.write('created prmtop of protein in solvent with ions successfully\n')
        return

    def createSolvatedComplex(self, padding=10):
        '''
        Method to generate protein+ligand in water prmtop and restart using the parameters generated for the ligand in the antechamberLigand method

        padding: extents of box padding to be fed to tleap solvateBox command, default here is 10 Å, wouldn't go too much larger
        '''
        boxstring = self.watFF[6:].upper() + 'BOX'
        os.system('rm -rf leap.log') #clean leap log so we can search info about these steps more easily afterwards
        with open('tleap5.in','w') as file:
            l1 = f"source leaprc.{self.proFF}\n"
            l2 = f"source leaprc.{self.watFF}\n"
            l3 = f'source leaprc.{self.ligFF}\n'
            l4 = 'loadamberparams LIG.frcmod\n'
            l5 = 'loadoff LIG.lib\n'
            l6 = f"protein = loadpdb {self.pdbFile}\n" 
            l7 = 'ligand = loadmol2 ligand.mol2\n'
            l8 = 'complex = combine{protein ligand}\n'
            l9 = f"solvatebox complex {boxstring} {padding}\n"
            l10 = "saveamberparm complex complex_wat.prmtop complex_wat.rst7\n"
            l11 = "quit"
            file.writelines([l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11])
        os.system("tleap -f tleap5.in")
        self.log.write('created prmtop of complex in solvent successfully\n')
        return

    def createIonsComplex(self, padding=10, targetM = 0.15):
        '''
        Method to generate a protein+ligand in water with counter ions (added to desired molarity), based on parameters generated in antechamberLigand method

        padding: extents of box padding to be fed to tleap solvateBox command, default here is 10 Å, wouldn't go too much larger
        targetM: target ion molarity (M), default 0.15, but if set to 0 will only add counter ions to neutralize charge of system
        '''
        boxstring = self.watFF[6:].upper() + 'BOX'
        ### first use leap.log from solvation of protein to determine system charge
        charge = 0
        with open('leap.log', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'The unperturbed charge of the unit' in line:
                    print(line)
                    splits = line.rsplit()
                    charge = int(float(splits[6][1:-1]))
                    print(int(float(splits[6][1:-1])))
        if charge == 0:
            context = 'uncharged'
        elif charge < 0:
            context = 'negative'
        elif charge > 0:
            context = 'positive'
        with open('complex_wat.rst7', 'r') as f:
            last_line = f.readlines()[-1]
        raw = last_line.rsplit()
        protein_wat_boxcoord = []
        for i in range(3):
            protein_wat_boxcoord.append(float(raw[i]))
        print(f'coordinates of box: {protein_wat_boxcoord}')
        protein_wat_boxvol = protein_wat_boxcoord[0]*protein_wat_boxcoord[1]*protein_wat_boxcoord[2]
        protein_wat_boxvol_L = protein_wat_boxvol * float(1/(10**10)**3) * float((10**2)**3/1) * float(1/10**3)
        target_molarity = 0.15 #M
        target_molarity_mmol = target_molarity*1000
        target_ion_atoms__L = target_molarity_mmol * float(1/10**3) * float(6.022*10**23)
        target_num_ions = int(protein_wat_boxvol_L * target_ion_atoms__L)
        print(f'volume of box (Å^2): {protein_wat_boxvol}')
        print(f'volume of box (L): {protein_wat_boxvol_L}')
        print(f'target ionic concentration (mmol): {target_molarity_mmol}')
        print(f'target number of ions (atoms/L): {target_ion_atoms__L}')
        if context == 'uncharged':
            print(f'target number of ions (atoms): {target_num_ions} Na, {target_num_ions} Cl')
            numNa = target_num_ions
            numCl = target_num_ions
        if context == 'negative':
            print(f'target number of ions (atoms): {target_num_ions+abs(charge)} Na, {target_num_ions} Cl')
            numNa = target_num_ions+abs(charge)
            numCl = target_num_ions
        if context == 'positive':
            print(f'target number of ions (atoms): {target_num_ions} Na, {target_num_ions+abs(charge)} Cl')
            numNa = target_num_ions
            numCl = target_num_ions+abs(charge)
        if targetM==0.0:
            #user has specified to add no additional ions, only counters
            if context == 'uncharged':
                numNa = 0
                numCl = 0
            elif context == 'negative':
                numNa = 0+abs(charge)
                numCl = 0
            elif context == 'positive':
                numNa = 0
                numCl = 0+abs(charge)
        #now leap
        with open('tleap6.in','w') as file:
            l1 = f"source leaprc.{self.proFF}\n"
            l2 = f"source leaprc.{self.watFF}\n"
            l3 = f'source leaprc.{self.ligFF}\n'
            l4 = 'loadamberparams LIG.frcmod\n'
            l5 = 'loadoff LIG.lib\n'
            l6 = f"protein = loadpdb {self.pdbFile}\n" 
            l7 = 'ligand = loadmol2 ligand.mol2\n'
            l8 = 'complex = combine{protein ligand}\n'
            l9 = f"solvatebox complex {boxstring} {padding}\n"
            l10 = f'addIonsRand complex Na+ {numNa} Cl- {numCl}\n'
            l11 = "saveamberparm complex complex_ion.prmtop complex_ion.rst7\n"
            l12 = "quit"
            file.writelines([l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12])
            
        ret = os.system("tleap -f tleap6.in")
        return

    def createAmberOpenMM(self):
        '''
        Method to import prmtops generated by tleap into an openMM system as desired by the sysType description by the user
            Be sure the initialization of simSystem is accurate to the desired simulation/analyses
        '''
        if self.sysType == 'P-L':
            if self.ions == None: #No ions whatsoever
                amber_structure=pmd.load_file("complex_wat.prmtop", "complex_wat.rst7")
                self.prmtop = openmm.app.AmberPrmtopFile('complex_wat.prmtop')
                self.inpcrd = openmm.app.AmberInpcrdFile('complex_wat.rst7')
            else: #ions, whether counter or to a molarity
                amber_structure=pmd.load_file("complex_ion.prmtop", "complex_ion.rst7")
                self.prmtop = openmm.app.AmberPrmtopFile('complex_ion.prmtop')
                self.inpcrd = openmm.app.AmberInpcrdFile('complex_ion.rst7')
        elif self.sysType == 'Apo':
            if self.ions == None: #No ions whatsoever
                amber_structure=pmd.load_file("protein_wat.prmtop", "protein_wat.rst7")
                self.prmtop = openmm.app.AmberPrmtopFile('protein_wat.prmtop')
                self.inpcrd = openmm.app.AmberInpcrdFile('protein_wat.rst7')
            else: #ions, whether counter or to a molarity
                amber_structure=pmd.load_file("protein_ion.prmtop", "protein_ion.rst7")
                self.prmtop = openmm.app.AmberPrmtopFile('protein_ion.prmtop')
                self.inpcrd = openmm.app.AmberInpcrdFile('protein_ion.rst7')        
        elif self.sysType == 'Lig':
            amber_structure=pmd.load_file("ligand_wat.prmtop", "ligand_wat.rst7")
            self.prmtop = openmm.app.AmberPrmtopFile('ligand_wat.prmtop')
            self.inpcrd = openmm.app.AmberInpcrdFile('ligand_wat.rst7')

        self.omm_system = amber_structure.createSystem(
            nonbondedMethod=openmm.app.PME,
            nonbondedCutoff=9.0 * openmm_unit.angstrom,
            switchDistance=10.0 * openmm_unit.angstrom,
            constaints=openmm.app.HBonds,
            removeCMMotion=False
        )
        integrator = openmm.LangevinIntegrator(
            300 * openmm_unit.kelvin,
            1 / openmm_unit.picosecond,
            0.002 * openmm_unit.picoseconds
        )
        simulation=openmm.app.Simulation(self.prmtop.topology, self.omm_system, integrator)
        self.omm_top = simulation.topology
        simulation.context.setPositions(self.inpcrd.positions)
        if self.inpcrd.boxVectors is not None:
            simulation.context.setPeriodicBoxVectors(*self.inpcrd.boxVectors)
        openmm.app.PDBFile.writeFile(simulation.topology, self.inpcrd.positions, open('topology.pdb', 'w'))
        # self.solvatedTop = Topology.from_pdb('topology.pdb', unique_molecules=[self.ligand])
        return
    
    
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
        simulation.context.setPositions(self.inpcrd.positions)
        if self.inpcrd.boxVectors is not None:
            simulation.context.setPeriodicBoxVectors(*self.inpcrd.boxVectors)
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
                      nc_reporter='prod.dcd',state_reporter='prod.csv',saveFreq=5000, simSaveFreq=500):
        '''
        Method for running production simulations in NVT systems
        
        temperature: simulation temperature, default 300 (K)
        mdsteps: number of md steps to be performed, default 50,000,000 (total 100 ns)
        frictCoeff: integrator frictional coefficient, default 1 (1/ps)
        stepSize: simulation step size, default 0.002 (ps)
        nc_reporter: trajectory file output, default 'prod.nc'
        state_reporter: trajectory log output, default 'prod.csv'
        saveFreq: frequency to write to the log, default 5000 (every 10 ps)
        simSaveFreq: frequency to write to the trajectory output file, default 500 (1 ps)
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
            simulation.system.getForce(5).setFrequency(0)
        else:
            simulation.system.getForce(6).setFrequency(0)
        nc_split = nc_reporter.rsplit('.')
        if nc_split[-1] == 'dcd':
            self.log.write(f'User specified DCD trajectory format: {nc_reporter}\n')
            nc_reporter = openmm.app.DCDReporter(nc_reporter, simSaveFreq)
        elif nc_split[-1] == 'nc':
            self.log.write(f'User specified NetCDF trajectory format: {nc_reporter}\n')
            nc_reporter = pmd.openmm.NetCDFReporter(nc_reporter, simSaveFreq)
        else:
            self.log.write(f'User specified a trajectory file format that is not recognized! {nc_reporter}\n')
            self.log.write('Please provide an output file with the correct extensions: either .nc or .dcd\n')
            return 1
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
