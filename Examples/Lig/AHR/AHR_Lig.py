from Interchange_Simulation import simSystem

sys_pdb = '5bt3_prepared.pdb'
lig_pdb = 'ligand_prepared.pdb'
lig_sdf = 'ligand.sdf'

system = simSystem(lig_pdb, 'Lig', lig_sdf, 'AHR')
system.solvateSystem(padding=2.0,ionicStrength=0.0)
system.embedMolecule()
system.createInterchange()
system.addRestraints()
system.minimizeSystem()
system.heatSystem()
system.equilSystem()
system.prodSimulation(mdsteps=5000000)
