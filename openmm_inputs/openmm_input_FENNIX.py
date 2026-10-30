import os, sys, time
from openmm import app, unit, Platform, LangevinMiddleIntegrator, MonteCarloBarostat
from openmmml import MLPotential
from parmed import load_file
from parmed.openmm import RestartReporter
import fennol

# Download model
# wget https://github.com/FeNNol-tools/FeNNol-PMC/raw/refs/heads/main/FENNIX-BIO1/fennix-bio1M.fnx

MODELPATH=os.path.expandvars("$SCRATCH/examples_mlmd/models/fennix-bio1M.fnx")

# Load AMBER topology and restart
prmtop="system.parm7"
restart="system.rst7"
amber_sys = load_file(prmtop, restart)
ncrst=app.amberinpcrdfile.AmberInpcrdFile(restart)

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

# Select atoms for the ML region
ml_atoms = [ ]
for atom in amber_sys.topology.atoms():
    if atom.residue.name in ("ACE", "ALA", "NME"):
        ml_atoms.append(atom.index)

# Create a hybrid ML/MM system using the FENNIX potential
potential = MLPotential('fennix', modelPath=MODELPATH)

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

# initialize simulation state from the Amber restart,
sim.context.setPositions(amber_sys.positions)
if ncrst.boxVectors is not None:
    sim.context.setPeriodicBoxVectors(*ncrst.boxVectors)
if ncrst.velocities is not None:
    sim.context.setVelocities(ncrst.velocities)
else:
    sim.context.setVelocitiesToTemperature(temperature)

# Add a reporter to print selected simulation statistics every 200 steps
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

