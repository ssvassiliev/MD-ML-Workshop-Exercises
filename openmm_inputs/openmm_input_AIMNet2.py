# Model: wget https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2/aimnet2_wb97m_d3_0.pt
# Create ~/.cache/aimnet and make a link to it:
# mkdir $HOME/.cache/aimnet
# ln -s $(pwd)/aimnet/aimnet2_wb97m_d3_0.pt ~/.cache/aimnet/aimnet2_wb97m_d3_0.pt 

import os, sys, time
from openmm import app, unit, Platform, LangevinMiddleIntegrator, MonteCarloBarostat
from openmmml import MLPotential
from parmed import load_file
from parmed.openmm import RestartReporter

# Load AMBER files
amber_sys = load_file("system.parm7", "system.rst7")
ncrst=app.amberinpcrdfile.AmberInpcrdFile("system.rst7")

# Simulation parameters
nsteps = 20000
timestep = 1.0 * unit.femtoseconds
temperature = 310 * unit.kelvin
pressure = 1 * unit.atmosphere

# Create base MM system
system = amber_sys.createSystem (
    nonbondedMethod = app.PME,
    ewaldErrorTolerance = 0.0004,
    nonbondedCutoff = 8.0 * unit.angstroms,
    constraints = app.HBonds,
    removeCMMotion = True
)

# Identify ML atoms
ml_atoms = [ ]
for atom in amber_sys.topology.atoms():
    if atom.residue.name in ("ACE", "ALA", "NME"):
        ml_atoms.append(atom.index)

# Create mixed system
potential = MLPotential('aimnet2')

system = potential.createMixedSystem(
    amber_sys.topology,
    system,
    ml_atoms,
    removeConstraints=False
)

integrator = LangevinMiddleIntegrator(temperature, 1.0 / unit.picoseconds, timestep)
barostat = MonteCarloBarostat(pressure, temperature, 50)
system.addForce(barostat)


platform = Platform.getPlatformByName("CUDA")
print("Platform:", platform.getName())
properties = dict(
    CudaPrecision = "mixed",
    UseCpuPme = 'false',
    DeterministicForces = "false",
    DeviceIndex = os.environ["CUDA_VISIBLE_DEVICES"],
)

sim = app.Simulation(amber_sys.topology, system, integrator, platform, properties)
sim.context.setPositions(amber_sys.positions)
sim.context.setVelocities(ncrst.velocities)

# Initialize velocities from a Maxwell–Boltzmann distribution at 310 K
# sim.context.setVelocitiesToTemperature(310 * unit.kelvin)

sim.reporters.append(app.StateDataReporter(
    sys.stdout, 
    1000,
    step = True, 
    potentialEnergy = True, 
    kineticEnergy = True,
    totalEnergy = True, 
    temperature = True, 
    volume = True
    )
)

sim.reporters.append(app.DCDReporter("trajectory.dcd", 50000))
sim.reporters.append(RestartReporter("restart.nc", 50000, netcdf = True))

print("\nRunning dynamics...")
start = time.time()

sim.step(nsteps)

elapsed = time.time() - start
print(f"Elapsed: {elapsed:.2f} sec")

# Benchmark time
simulated_ns = (nsteps * timestep).value_in_unit(unit.nanoseconds)
ns_per_day = simulated_ns / (elapsed / 86400)

print(f"Performance: {ns_per_day:.2f} ns/day")

