from Interchange_Simulation import simSystem

sys_pdb = '5bt3_prepared.pdb'
lig_pdb = 'ligand.pdb'
lig_sdf = 'ligand.sdf'

system = simSystem(sys_pdb, 'P-L', lig_sdf)
system.solvateSystem()
system.embedMolecule()
system.createInterchange()
system.minimizeSystem()
system.heatSystem()
system.equilSystem()
system.prodSimulation(mdsteps=5000000)
