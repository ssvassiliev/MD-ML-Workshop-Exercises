import os, sys, time
from openmm import app, unit, Platform, LangevinMiddleIntegrator, MonteCarloBarostat
from openmmml import MLPotential
from parmed import load_file
from parmed.openmm import RestartReporter

# Load AMBER topology and restart
amber_sys = load_file("system.parm7", "system.rst7")
ncrst=app.amberinpcrdfile.AmberInpcrdFile("system.rst7")

# Simulation parameters
nsteps = 20000
timestep = 1.0 * unit.femtoseconds
temperature = 310 * unit.kelvin
pressure = 1 * unit.atmosphere

# Create the base MM system
# This establishes the original force field for the entire system, including 
# bonded terms, nonbonded interactions, constraints, and PME settings.
system = amber_sys.createSystem (
    nonbondedMethod = app.PME,
    ewaldErrorTolerance = 0.0004,
    nonbondedCutoff = 8.0 * unit.angstroms,
    constraints = app.HBonds,
    removeCMMotion = True
)

# FIX ME 1: Select atoms belonging to the ML region
# 1. Create an empty list: ml_atoms = []
# 2. Loop over atoms in amber_sys.topology.atoms()
# 3. Check whether each atom's residue name is "ACE", "ALA", or "NME"
# 4. If it is, append the atom's index (atom.index) to ml_atoms
# for atom in ???:
#     if atom.residue.name in ( ??? ):
#         ml_atoms.append( ??? )

# FIX ME 2: Initialize the ANI-2x ML potential
# Create an MLPotential object using the model name 'ani2x'
# potential = ???

# FIX ME 3: Create a hybrid ML/MM system using the ani2x potential
# Use potential.createMixedSystem() with:
# - The AMBER topology (amber_sys.topology)
# - The existing MM system (system)
# - The list of ML atom indices (ml_atoms)
# - removeConstraints=False
# Assign the returned hybrid system to system
# system = ???

# OpenMM-ML uses the existing MM system as its starting point and replaces 
# the appropriate force contributions for the selected atoms.

# Configure the Langevin integrator and barostat
integrator = LangevinMiddleIntegrator(temperature, 1.0 / unit.picoseconds, timestep)
barostat = MonteCarloBarostat(pressure, temperature, 50)
system.addForce(barostat)

# Select the GPU implementation
platform = Platform.getPlatformByName("CUDA")
print("Platform:", platform.getName())
properties = dict(
    CudaPrecision = "mixed",
    UseCpuPme = 'false',
    DeterministicForces = "false",
    DeviceIndex = os.environ["CUDA_VISIBLE_DEVICES"],
)

# Create the simulation
sim = app.Simulation(amber_sys.topology, system, integrator, platform, properties)

# initialize simulation state from the Amber restart,
sim.context.setPositions(amber_sys.positions)
if ncrst.boxVectors is not None:
    sim.context.setPeriodicBoxVectors(*ncrst.boxVectors)
if ncrst.velocities is not None:
    sim.context.setVelocities(ncrst.velocities)
else:
    sim.context.setVelocitiesToTemperature(temperature)

# Add a reporter to print selected simulation statistics every 1000 steps
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

# Add a reporter to save the trajectory 
# sim.reporters.append(app.DCDReporter("trajectory.dcd", 1000))

# Add a reporter to save the restart
# sim.reporters.append(RestartReporter("restart.nc", 10000, netcdf=True))

sim.step(100)  # Warmup 

print("\nRunning dynamics...")
start = time.time()

sim.step(nsteps)

elapsed = time.time() - start
print(f"Elapsed: {elapsed:.2f} sec")

# Calculate simulation performance
simulated_ns = (nsteps * timestep).value_in_unit(unit.nanoseconds)
ns_per_day = simulated_ns / (elapsed / 86400)

print(f"Performance: {ns_per_day:.2f} ns/day")

