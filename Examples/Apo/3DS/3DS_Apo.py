from Interchange_Simulation import simSystem

sys_pdb = '5bt3_prepared.pdb'
lig_pdb = 'ligand.pdb'
lig_sdf = 'ligand.sdf'

system = simSystem(pdbFile=sys_pdb, sysType='Apo', sdfFile=lig_sdf, restraintType='3DS')
system.solvateSystem()
system.embedMolecule()
system.createInterchange()
system.addRestraints(atomList=['856','1195','1710'])
system.minimizeSystem()
system.heatSystem()
system.releaseSystem()
system.equilSystem()
system.prodSimulation(mdsteps=5000000)
