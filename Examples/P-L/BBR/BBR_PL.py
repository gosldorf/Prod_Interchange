from Interchange_Simulation import simSystem

sys_pdb = '5bt3_prepared.pdb'
lig_pdb = 'ligand.pdb'
lig_sdf = 'ligand.sdf'

system = simSystem(pdbFile=sys_pdb, sysType='P-L', sdfFile=lig_sdf, restraintType='BBR')
system.solvateSystem()
system.embedMolecule()
system.createInterchange()
system.addRestraints()
system.minimizeSystem()
system.heatSystem()
system.releaseSystem()
system.equilSystem()
system.prodSimulation(mdsteps=5000000)
