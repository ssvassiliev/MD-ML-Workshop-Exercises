import os, sys, time
import torch
from openmm import app, unit, Platform, LangevinMiddleIntegrator, MonteCarloBarostat
from openmmml import MLPotential
from parmed import load_file
from parmed.openmm import RestartReporter

# Load AMBER topology and restart
prmtop = "system.parm7"
restart = "system.rst7"
amber_sys = load_file(prmtop, restart)
ncrst = app.AmberInpcrdFile(restart)

# Simulation parameters
nsteps = 20000
timestep = 1.0 * unit.femtoseconds
temperature = 310 * unit.kelvin
pressure = 1 * unit.atmosphere

# Create base MM system
system = amber_sys.createSystem(
    nonbondedMethod=app.PME,
    ewaldErrorTolerance=0.0004,
    nonbondedCutoff=8.0 * unit.angstroms,
    constraints=app.HBonds,
    removeCMMotion=True,
)

# Select atoms for the ML region
ml_atoms = [a.index for a in amber_sys.topology.atoms()
            if a.residue.name in ("ACE", "ALA", "NME")]

# Create a hybrid ML/MM system using the orb-v3-omol potential
# Model is downloaded into ~/.cache/cached_path
# To use local weights only
# LOCAL_WEIGHTS = "<path_to>/orb-v3-conservative-omol-20250820.ckpt"
# orb_pretrained.cached_path = lambda *args, **kwargs: LOCAL_WEIGHTS

potential = MLPotential("orb-v3-conservative-omol")
system = potential.createMixedSystem(
    amber_sys.topology, system, ml_atoms, removeConstraints=False
)



# Enable TensorFloat-32 tensor cores
torch.set_float32_matmul_precision("high")

# Configure the Langevin integrator and barostat
integrator = LangevinMiddleIntegrator(temperature, 1.0 / unit.picoseconds, timestep)
system.addForce(MonteCarloBarostat(pressure, temperature, 50))

# Select the GPU implementation
platform = Platform.getPlatformByName("CUDA")
properties = dict(
    CudaPrecision="mixed",
    UseCpuPme="false",
    DeterministicForces="false",
    DeviceIndex=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
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

print("\nRunning dynamics...\n")
start = time.time()

sim.step(nsteps)

elapsed = time.time() - start
print(f"Elapsed: {elapsed:.2f} sec")

# Calculate simulation performance
simulated_ns = (nsteps * timestep).value_in_unit(unit.nanoseconds)
print(f"Performance: {simulated_ns / (elapsed / 86400):.2f} ns/day")
