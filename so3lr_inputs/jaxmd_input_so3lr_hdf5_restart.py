#!/usr/bin/env python
"""NPT MD of an AMBER-prepared system with SO3LR + jax_md (Nose-Hoover NPT)."""
import jax
import jax.numpy as jnp
import jax_md
import numpy as np
import pathlib
import parmed as pmd

from ase import Atoms
from ase.geometry import cellpar_to_cell
from ase.io import read
from jax_md import units                # jax_md unit systems
from typing import Dict

from so3lr import to_jax_md
from so3lr import So3lrPotential

import time
import h5py

# For float64 simulations, call this before any JAX computation:
# jax.config.update('jax_enable_x64', True)

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
prmtop='system.parm7'
restrt='system.rst7'
hdf5_restart = 'minimized.hdf5'
energy_file = 'energies.dat'

unit = units.metal_unit_system() # jax_md uses metal system: eV, A, ps, K

timestep_ps = 0.001   # MD time step [ps]
npt_cycles = 100      # Number of cycles (Python-level loop iterations)
npt_steps = 100       # MD steps per cycle (one compiled fori_loop).
                      # Total MD steps = npt_cycles * npt_steps

T_init = 300          # Target temperature [K] (also used for initial velocities)
pressure = 1.         # Target pressure [bar] 

# --- Convert to JAX-MD units ---
timestep = timestep_ps * unit['time']        # jax_md internal time
T_init = T_init * unit['temperature']        # kT in eV 
pressure=pressure * unit['pressure']         # pressure in eV/A^3 

# Nose-Hoover chain settings (shared by thermostat and barostat).
chain = 3             # Chain length
chain_steps = 2       # Steps per chain
sy_steps = 3          # Suzuki-Yoshida steps

# Coupling time constants, as multiples of the time step.
thermo = 100          # Thermostat
baro = 1000           # Barostat

# Calculate thermostat and barostat coupling times
thermostat_tau = thermo * timestep
barostat_tau = baro * timestep

def load_restart_from_hdf5(h5_file, frame=-1):
    with h5py.File(h5_file, 'r') as f:
        n = f['positions'].shape[-2]
        R = f['positions'][:].reshape(-1, n, 3)[frame].astype(np.float64)    # [A]
        V = f['velocities'][:].reshape(-1, n, 3)[frame].astype(np.float64)   # [A/ps]
        edge = float(f['box'][:].reshape(-1, f['box'].shape[-1])[frame][0])  # [A]
    return R, V, np.array([edge, edge, edge, 90., 90., 90.])

# ---------------------------------------------------------------------------
# System setup: read the HDF5 dump
# ---------------------------------------------------------------------------
struct = pmd.load_file(prmtop)

# Drop virtual sites (e.g. OPC "EPW", atomic number 0): SO3LR only knows real atoms.
# The same mask must be applied to every per-atom array (positions, velocities, ...).
keep = np.array([a.atomic_number > 0 for a in struct.atoms])
print("removing", (~keep).sum(), "virtual sites")
numbers = np.array([a.atomic_number for a in struct.atoms])[keep]

R, V, cellpar = load_restart_from_hdf5(hdf5_restart) 

atoms = Atoms(
    numbers=numbers,
    positions=R, 
)

# Periodic box: (a, b, c, alpha, beta, gamma) -> 3x3 cell
if cellpar is not None:
    atoms.set_cell(cellpar_to_cell(cellpar))
    atoms.set_pbc(True)
    atoms.wrap()
if V is not None:
    atoms.set_velocities(V)    # A/ps -> ASE units

species = jnp.array(atoms.get_atomic_numbers())
masses = jnp.array(atoms.get_masses())    # [amu]; use topology masses if HMR was used
num_atoms = len(species)

# The box must be a 1-element array (cubic cell, edge length in A).
box = jnp.array([np.array(atoms.get_cell())[0, 0]])

# Fractional coordinates are required for NPT (the box changes during the run).
fractional_coordinates = True
displacement, shift = jax_md.space.periodic_general(box=box, fractional_coordinates=fractional_coordinates)

positions_init = jnp.array(atoms.get_positions())
if fractional_coordinates:
    positions_init = positions_init / box

# ---------------------------------------------------------------------------
# SO3LR energy function and neighbor lists
# ---------------------------------------------------------------------------
# Returns separate short-range (sr) and long-range (lr) neighbor-list functions.
neighbor_fn, neighbor_fn_lr, energy_fn = to_jax_md(
    potential=So3lrPotential(
        dtype=jnp.float64  # or float32 for single precision
    ),
    displacement_or_metric=displacement,
    box_size=box,
    species=species,
    capacity_multiplier=1.3,            # extra neighbor-list capacity (fewer overflows)
    buffer_size_multiplier_sr=1.25,
    buffer_size_multiplier_lr=1.25,
    minimum_cell_size_multiplier_sr=1.0,
    disable_cell_list=False,
    fractional_coordinates=fractional_coordinates
)

energy_fn = jax.jit(energy_fn)
force_fn = jax.jit(jax_md.quantity.force(energy_fn))

nbrs = neighbor_fn.allocate(
    positions_init,
    box=box
)
nbrs_lr = neighbor_fn_lr.allocate(
    positions_init,
    box=box
)

# ---------------------------------------------------------------------------
# NPT integrator (Nose-Hoover)
# ---------------------------------------------------------------------------

# Shared chain settings, with separate coupling times
new_nhc_kwargs = {
    'chain_length': chain,
    'chain_steps': chain_steps,
    'sy_steps': sy_steps
}
thermostat_kwargs = {
    **new_nhc_kwargs,
    "tau": thermostat_tau,
}
barostat_kwargs = {
    **new_nhc_kwargs,
    "tau": barostat_tau,
}

rng_key = jax.random.PRNGKey(0)

# Create the NPT integrator
# init_fn initializes the simulation state
# apply_fn advances the simulation by one MD step
init_fn, apply_fn = jax_md.simulate.npt_nose_hoover(
    energy_fn,
    shift,
    dt=timestep,
    pressure=pressure,
    kT=T_init,
    barostat_kwargs=barostat_kwargs,
    thermostat_kwargs=thermostat_kwargs
)

# JIT-compile the functions
apply_fn = jax.jit(apply_fn)
init_fn = jax.jit(init_fn)

# Initialize from the initial positions; this draws Maxwell-Boltzmann momenta at kT.
state = init_fn(
    rng_key,
    positions_init,
    box=box,
    neighbor=nbrs.idx,
    neighbor_lr=nbrs_lr.idx,
    kT=T_init,
    mass=masses
)

# Optionally replace momenta by the restart velocities. 
if V is not None and np.any(V != 0.0):
    state = state.set(momentum=masses[:, None] * jnp.array(V))   # p = m * v, V in A/ps
    print("Using velocities from file.")
else:
    print("Velocities are zero or missing, keeping Maxwell-Boltzmann momenta from init_fn.")

# Check the initial temperature
T0 = jax_md.quantity.temperature(momentum=state.momentum, mass=state.mass) / unit['temperature']
print("Measured T after initialization [K]:", T0)


@jax.jit
def step_npt_fn(i, state):
    """One NPT step (thermostat + barostat act inside apply_fn), then refresh neighbors."""
    state, nbrs, nbrs_lr, box = state

    state = apply_fn(
        state,
        neighbor=nbrs.idx,
        neighbor_lr=nbrs_lr.idx,
        kT=T_init,
        pressure=pressure
    )

    # The box is a dynamical variable and changes every step.
    box = jax_md.simulate.npt_box(state)

    nbrs = nbrs.update(
        state.position,
        neighbor=nbrs.idx,
        box=box
    )

    nbrs_lr = nbrs_lr.update(
        state.position,
        neighbor_lr=nbrs_lr.idx,
        box=box
    )

    return state, nbrs, nbrs_lr, box


@jax.jit
def pressure_bar(state, nbrs, nbrs_lr, box):
    """Instantaneous pressure [bar] of a cubic box: P = (2*KE - dE/d_eps) / (3*V).

    dE/d_eps is the energy derivative under isotropic strain (box scaled by 1 + eps);
    together with the kinetic term this is the virial pressure.
    """
    def energy_scaled(eps):
        pos = state.position if fractional_coordinates else state.position * (1.0 + eps)
        return energy_fn(
            pos,
            neighbor=nbrs.idx,
            neighbor_lr=nbrs_lr.idx,
            box=box * (1.0 + eps)
        )

    dE_deps = jax.grad(energy_scaled)(jnp.zeros((), dtype=box.dtype))
    KE = jax_md.quantity.kinetic_energy(momentum=state.momentum, mass=state.mass)
    volume = box[0] ** 3                              # [A^3], cubic box
    return (2.0 * KE - dE_deps) / (3.0 * volume) / unit['pressure'] 


# ---------------------------------------------------------------------------
# Trajectory output (HDF5)
# ---------------------------------------------------------------------------
# Frames are collected in memory and flushed to the HDF5 file `batch_size` frames
# at a time, which is much faster than per-frame ase.io.write for large systems.
# Each dataset needs a fixed shape up front, e.g. positions: (batch_size, num_atoms, 3).
# After k flushes the file holds arrays of shape (k, batch_size, ...).
from mlff.mdx.hdfdict import DataSetEntry, HDF5Store

def init_hdf5_store(
    save_to,
    batch_size,
    num_atoms,
    num_box_entries,
    exist_ok=True
):

    parent_dir = pathlib.Path(save_to).expanduser().resolve().parent
    parent_dir.mkdir(exist_ok=True)

    _save_to = pathlib.Path(save_to)
    if _save_to.exists():
        if exist_ok is False:
            raise RuntimeError(
                f'File exists save_to={_save_to}. '
                f'Set exists_ok=True to override file.'
            )

    dataset = {
        'positions': DataSetEntry(
            chunk_length=1,
            shape=(batch_size, num_atoms, 3),
            dtype=np.float32
        ),
        'velocities': DataSetEntry(
            chunk_length=1,
            shape=(batch_size, num_atoms, 3),
            dtype=np.float32
        ),
        'box': DataSetEntry(
            chunk_length=1,
            shape=(batch_size, num_box_entries),
            dtype=np.float32
        )
    }

    return HDF5Store(_save_to, datasets=dataset, mode='w')


# Frames collected before each flush (100 is a good default for long runs).
# One frame is stored per cycle, so frame spacing = npt_steps * timestep_ps.
hdf5_buffer_size = 3

hdf5_store = init_hdf5_store(
    save_to='hdf5/trajectory.hdf5',
    batch_size=hdf5_buffer_size,
    num_atoms=num_atoms,
    num_box_entries=1,   # cubic box: one edge length [A]
    exist_ok=True        # overwrite an existing file
)

# ---------------------------------------------------------------------------
# Production run
# ---------------------------------------------------------------------------
total_time = time.time()

velocities, positions, boxes = [], [], []

header = (
    "{:>4}{:>12}{:>12}{:>12}{:>10}{:>12}{:>10}{:>10}{:>10}"
).format(
    "Step", "KE [eV]", "PE [eV]", "Tot [eV]",
    "T [K]", "P [bar]", "L [A]", "rho", "s/step"
)

pathlib.Path(energy_file).expanduser().resolve().parent.mkdir(exist_ok=True)
efile = open(energy_file, 'w')
efile.write('# ' + header + '\n')   # '#' so np.loadtxt skips it
efile.flush()
print(f"\nRunning dynamics..\n\n{header}\n{'-' * len(header)}")

for i in range(npt_cycles):

    old_time = time.time()

    # Run `npt_steps` NPT steps on the device. (Thermostat/barostat act every step.)
    new_state, nbrs, nbrs_lr, new_box = jax.block_until_ready(
        jax.lax.fori_loop(
            0,
            npt_steps,
            step_npt_fn,
            (state, nbrs, nbrs_lr, box)
        )
    )

    new_time = time.time()

    # If a neighbor list overflowed, the block's result is invalid: reallocate and
    # redo the cycle from the previous state/box (they are only updated in `else`).
    if nbrs.did_buffer_overflow:
        print('Neighbor list overflowed, reallocating.')
        nbrs = neighbor_fn.allocate(state.position, box = box)
        if nbrs_lr.did_buffer_overflow:
            print('Long-range neighbor list also overflowed, reallocating.')
            nbrs_lr = neighbor_fn_lr.allocate(state.position, box = box)
    elif nbrs_lr.did_buffer_overflow:
        print('Long-range neighbor list overflowed, reallocating.')
        nbrs_lr = neighbor_fn_lr.allocate(state.position, box = box)
    else:
        state = new_state
        box = new_box

        # Only save frames from cycles without overflows.

        ######## HDF5 saving part ########

        # Velocities from momenta [A/ps].
        velocities.append(
            jnp.divide(
                state.momentum,
                state.mass
            )
        )

        # Last frame of the cycle, converted from fractional to Cartesian [A].
        positions.append(
            jax_md.space.transform(box=box, R=state.position)  # only correct for cubic cells with equally long lattice vectors
        )

        boxes.append(
            box
        )

    # Flush once the buffer holds `hdf5_buffer_size` frames.
    if (len(positions) % hdf5_buffer_size == 0) and (len(positions) > 0):

        step_data = dict(
            velocities=jnp.stack(velocities, axis=0),  # (hdf5_buffer_size, num_atoms, 3)
            positions=jnp.stack(positions, axis=0),    # (hdf5_buffer_size, num_atoms, 3)
            box=jnp.stack(boxes, axis=0)               # (hdf5_buffer_size, num_box_entries)
        )

        # jax arrays -> numpy arrays for h5py
        step_data = jax.tree.map(lambda u: np.asarray(u), step_data)

        hdf5_store.append(
            step_data
        )

        velocities, positions, boxes = [], [], []

        ######## End of HDF5 saving part ########

    # Thermodynamic output (energies in eV, T in K, P in bar)
    KE = jax_md.quantity.kinetic_energy(
        momentum=state.momentum,
        mass=state.mass
    )

    PE = energy_fn(
        state.position,
        neighbor=nbrs.idx,
        neighbor_lr=nbrs_lr.idx,
        box=box
    )

    T = jax_md.quantity.temperature(
        momentum=state.momentum,
        mass = state.mass
    ) / unit['temperature']

    P = pressure_bar(state, nbrs, nbrs_lr, box)
    L = float(box[0]) 
    rho = float(masses.sum()) * 1.66054 / L**3

    efile.write(
        f"{(i+1)*npt_steps:6d}"
        f"{float(KE):12.4f}"
        f"{float(PE):12.4f}"
        f"{float(KE+PE):12.4f}"
        f"{float(T):10.2f}"
        f"{float(P):12.2f}"
        f"{L:10.4f}"
        f"{rho:10.4f}"
        f"{(new_time - old_time) / npt_steps:10.4f}\n"
    )
    efile.flush()

wall = time.time() - total_time
print('Total_time: ', wall)
efile.close()

# ns/day over the whole run (includes JIT compilation, IO and any overflow redos).
total_steps = npt_cycles * npt_steps
ns_total = total_steps * timestep_ps * 1e-3                  # ps -> ns
print(f'Overall (incl. compile, IO, overflows): '
      f'{ns_total / wall * 86400:.3f} ns/day')

jax.clear_caches()

# ---------------------------------------------------------------------------
# Postprocess the trajectory
# ---------------------------------------------------------------------------

file = h5py.File('hdf5/trajectory.hdf5', mode='r')

# Stored shape: (num_flushes, hdf5_buffer_size, num_atoms, 3)
velocities = np.array(file['velocities'])
print(velocities.shape)

# Merge the two leading axes into one list of consecutive frames,
# separated by delta_t = npt_steps * timestep_ps [ps].
velocities = velocities.reshape(-1, *file['velocities'].shape[-2:])

print(velocities.shape)  # (num_flushes * hdf5_buffer_size, num_atoms, 3)
