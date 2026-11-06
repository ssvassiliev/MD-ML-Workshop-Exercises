import os, sys, time
from openmm import app, unit, Platform, LangevinMiddleIntegrator, MonteCarloBarostat
from openmmml import MLPotential
from parmed import load_file
from parmed.openmm import RestartReporter
import fennol

# Load AMBER topology and restart
prmtop="system.parm7"
restart="system.rst7"
energy_file = 'energies.dat'

amber_sys = load_file(prmtop, restart)
ncrst=app.amberinpcrdfile.AmberInpcrdFile(restart)

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

# Select atoms for the ML region
ml_atoms = [ ]
for atom in amber_sys.topology.atoms():
    if atom.residue.name in ("ACE", "ALA", "NME"):
        ml_atoms.append(atom.index)

# Create a hybrid ML/MM system using the FENNIX potential
potential = MLPotential('fennix-bio1-medium')

# Offline model loading:
# Download the FeNNix-Bio1M model file before submitting the job.
# Create the model directory
# mkdir -p $SCRATCH/examples_mlmd/models
# Download the FeNNix-Bio1M model
# wget -O $SCRATCH/examples_mlmd/models/fennix-bio1M.fnx \
#  https://github.com/FeNNol-tools/FeNNol-PMC/raw/refs/heads/main/FENNIX-BIO1/fennix-bio1M.fnx
# Specify its local path using modelPath 
# MODELPATH=os.path.expandvars("$SCRATCH/examples_mlmd/models/fennix-bio1M.fnx")
# potential = MLPotential('fennix', modelPath=MODELPATH)

system = potential.createMixedSystem(
    amber_sys.topology,
    system,
    ml_atoms,
    removeConstraints=False
)

# Configure the Langevin integrator and barostat
integrator = LangevinMiddleIntegrator(temperature, 1.0 / unit.picoseconds, timestep)
system.addForce(MonteCarloBarostat(pressure, temperature, 50))

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

# Initialize the simulation state using coordinates from the AMBER restart file.
sim.context.setPositions(amber_sys.positions)

# Restore periodic box dimensions if available (required for periodic simulations).
if ncrst.boxVectors is not None:
    sim.context.setPeriodicBoxVectors(*ncrst.boxVectors)

# Restore velocities from the AMBER restart file if available.
# Otherwise, generate new velocities from the target temperature.
if ncrst.velocities is not None:
    sim.context.setVelocities(ncrst.velocities)
else:
    sim.context.setVelocitiesToTemperature(temperature)
 
# Add a reporter to print selected simulation statistics every 200 steps
sim.reporters.append(app.StateDataReporter(
    energy_file, 
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
# sim.reporters.append(app.DCDReporter("trajectory.dcd", 5000))

# Add a reporter to save the restart
# sim.reporters.append(RestartReporter("restart.nc", 5000, netcdf = True))

sim.step(100)  # Warmup 

print("\nRunning dynamics...\n")
start = time.time()

sim.step(nsteps)

elapsed = time.time() - start
print(f"Elapsed: {elapsed:.2f} sec")

# Calculate simulation performance
simulated_ns = (nsteps * timestep).value_in_unit(unit.nanoseconds)
print(f"Performance: {simulated_ns / (elapsed / 86400):.2f} ns/day")

