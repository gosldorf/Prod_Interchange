import Interchange_Simulation

sys_pdb = '5bt3_prepared.pdb'
lig_pdb = 'ligand.pdb'
lig_sdf = 'ligand.sdf'

system = simSystem(sys_pdb, 'Apo', lig_sdf, 'AHR')
system.solvateSystem()
system.embedMolecule()
system.createInterchange()
system.minimizeSystem()
system.heatSystem()
system.releaseSystem()
system.equilSystem()
system.prodSimulation(mdsteps=5000000)