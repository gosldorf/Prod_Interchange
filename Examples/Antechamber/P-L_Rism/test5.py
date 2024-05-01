from Antechamber_Simulation import simSystem

sys_pdb = '6wgx_prepared.pdb'
lig_pdb = 'ligand_prepared.pdb'
lig_sdf = 'ligand.sdf'

system = simSystem(pdbFile=sys_pdb, sysType='P-L', ligFile=lig_pdb, sdfFile= lig_sdf, restraintType=None)
system.antechamberLigand()
system.createVacuumComplex()
