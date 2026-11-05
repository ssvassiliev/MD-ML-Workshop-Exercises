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

# For float64 simulations, call this before any JAX computation:
#jax.config.update('jax_enable_x64', True)

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
timestep_ps = 0.001   # MD time step [ps]
npt_cycles = 100      # Number of cycles (Python-level loop iterations)
npt_steps = 100       # MD steps per cycle (one compiled fori_loop).
                      # Total MD steps = npt_cycles * npt_steps

T_init = 300          # Target temperature [K] (also used for initial velocities)
pressure = 1.         # Target pressure, passed to jax_md as-is, i.e. in jax_md
                      # internal units (eV/A^3 for the metal system), NOT bar.
                      # 1 bar ~ 6.24e-7 eV/A^3, i.e. 1 bar = 1 * ase_units.bar.

# jax_md metal units: eV, A, ps, K. Convert time step, temperature and ... to them.
unit=units.metal_unit_system()

pressure=pressure * unit['pressure']

# Nose-Hoover chain settings (shared by thermostat and barostat).
chain = 3             # Chain length
chain_steps = 2       # Steps per chain
sy_steps = 3          # Suzuki-Yoshida steps

# Coupling time constants, as multiples of the time step.
thermo = 100          # Thermostat
baro = 1000           # Barostat


def default_nhc_kwargs(
    tau: jnp.float32,
    overrides: Dict
) -> Dict:
    """Nose-Hoover chain kwargs with defaults, overridden by `overrides`."""

    default_kwargs = {
        'chain_length': 3,
        'chain_steps': 2,
        'sy_steps': 3,
        'tau': tau
    }

    if overrides is None:
        return default_kwargs

    return {
      k: overrides.get(k, default_kwargs[k]) for k in default_kwargs
    }


# ---------------------------------------------------------------------------
# System setup: read the AMBER topology + restart
# ---------------------------------------------------------------------------
struct = pmd.load_file('4/polyAla_4.prmtop', '4/polyAla_4-relaxed2.rst7')

# Drop virtual sites (e.g. OPC "EPW", atomic number 0): SO3LR only knows real atoms.
# The same mask must be applied to every per-atom array (positions, velocities, ...).
keep = np.array([a.atomic_number > 0 for a in struct.atoms])
print("removing", (~keep).sum(), "virtual sites")

atoms = Atoms(
    numbers=np.array([a.atomic_number for a in struct.atoms])[keep],
    positions=struct.coordinates[keep],   # [A]
)

# Periodic box: (a, b, c, alpha, beta, gamma) -> 3x3 cell
if struct.box is not None:
    atoms.set_cell(cellpar_to_cell(struct.box))
    atoms.set_pbc(True)
    atoms.wrap()

# Velocities from the restart file (None if the restart has none).
# ParmEd gives A/ps; ASE expects A / (ASE time unit), hence the conversion.
vel = getattr(struct, 'velocities', None)
if vel is not None:
    vel = np.asarray(vel).reshape(-1, 3)[keep]
    atoms.set_velocities(vel / unit['time'])

species = jnp.array(atoms.get_atomic_numbers())
masses = jnp.array(atoms.get_masses())    # [amu]; use topology masses if HMR was used
num_atoms = len(species)

# The box must be a 1-element array (cubic cell, edge length in A). A 0-dim scalar
# makes jax_md.simulate.npt_box(state) return a 3x3 array, which leads to a
# TracerConversionError. Maybe open an issue on jax_md?
box = jnp.array(
    [
        np.array(atoms.get_cell())[0, 0]
    ]
)

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
    capacity_multiplier=1.25,            # extra neighbor-list capacity (fewer overflows)
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
# Energy minimization (FIRE)
# ---------------------------------------------------------------------------
# For repeated execution in a notebook, delete fire_state first; otherwise jit can
# meddle with things.
try:
    del fire_state
except NameError:
    pass

min_cycles = 5       # Python-level cycles (energy/force printed after each)
min_steps = 10       # FIRE steps per cycle

fire_init, fire_apply = jax_md.minimize.fire_descent(
    energy_fn,
    shift,
    dt_start = 0.05,
    dt_max = 0.1,
    n_min = 2
)

fire_apply = jax.jit(fire_apply)
fire_state = fire_init(
    positions_init,
    box=box,
    neighbor=nbrs.idx,
    neighbor_lr=nbrs_lr.idx
)

@jax.jit
def step_fire_fn(i, fire_state):
    """One FIRE step, then refresh both neighbor lists."""

    fire_state, nbrs, nbrs_lr = fire_state

    fire_state = fire_apply(
        fire_state,
        neighbor=nbrs.idx,
        neighbor_lr=nbrs_lr.idx,
        box=box
    )

    nbrs = nbrs.update(
        fire_state.position,
        neighbor=nbrs.idx,
        box=box
    )

    nbrs_lr = nbrs_lr.update(
        fire_state.position,
        neighbor_lr=nbrs_lr.idx,
        box=box
    )

    return fire_state, nbrs, nbrs_lr

print('Step\tE\tFmax')
print('--------------------------')
for i in range(min_cycles):
    fire_state, nbrs, nbrs_lr = jax.lax.fori_loop(
        0,
        min_steps,
        step_fire_fn,
        (fire_state, nbrs, nbrs_lr)
    )

    E = energy_fn(
        fire_state.position,
        neighbor=nbrs.idx,
        neighbor_lr=nbrs_lr.idx,
        box=box
    )

    F = force_fn(
        fire_state.position,
        neighbor=nbrs.idx,
        neighbor_lr=nbrs_lr.idx,
        box=box
    )

    print('{}\t{:.2f}\t{:.2f}'.format(i, E, np.abs(F).max()))

# ---------------------------------------------------------------------------
# NPT integrator (Nose-Hoover)
# ---------------------------------------------------------------------------
new_nhc_kwargs = {
    'chain_length': chain,
    'chain_steps': chain_steps,
    'sy_steps': sy_steps
}

timestep = timestep_ps * unit['time']        # jax_md internal time
T_init = T_init * unit['temperature']        # kT in eV (from here on T_init is an energy)

rng_key = jax.random.PRNGKey(0)

init_fn, apply_fn = jax_md.simulate.npt_nose_hoover(
    energy_fn,
    shift,
    dt=timestep,
    pressure=pressure,
    kT=T_init,
    barostat_kwargs=default_nhc_kwargs(baro * timestep, new_nhc_kwargs),
    thermostat_kwargs=default_nhc_kwargs(thermo * timestep, new_nhc_kwargs)
)

apply_fn = jax.jit(apply_fn)
init_fn = jax.jit(init_fn)

# Initialize from the relaxed positions; this draws Maxwell-Boltzmann momenta at kT.
state = init_fn(
    rng_key,
    fire_state.position,
    box=box,
    neighbor=nbrs.idx,
    neighbor_lr=nbrs_lr.idx,
    kT=T_init,
    mass=masses
)

# Optionally replace them by the restart velocities. Only do this if the restart
# actually contained velocities; otherwise atoms.get_momenta() is all zeros.
if vel is not None:
    state = state.set(momentum=jnp.array(atoms.get_momenta()))

print("kT passed to init_fn [eV]:", T_init)
T0 = jax_md.quantity.temperature(momentum=state.momentum, mass=state.mass) / unit['temperature']
print("measured T after init [K]:", T0)


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
    return (2.0 * KE - dE_deps) / (3.0 * volume) / unit['pressure']   # eV/A^3 -> bar


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
print('Step\tKE[eV]\tPE[eV]\tTot[eV]\tT[K]\tP[bar]\tL\trho\ts/step')
print('-------------------------------------------------------------------------')
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

    print(
        f'{(i+1)*npt_steps}\t{KE:.2f}\t{PE:.2f}\t{KE+PE:.3f}\t{T:.1f}\t{P:.1f}\t{L:.3f}\t{rho:.4f}\t{(new_time - old_time) / npt_steps:.4f}'
    )

wall = time.time() - total_time
print('Total_time: ', wall)

# ns/day over the whole run (includes JIT compilation, IO and any overflow redos).
total_steps = npt_cycles * npt_steps
ns_total = total_steps * timestep_ps * 1e-3                  # ps -> ns
print(f'Overall (incl. compile, IO, overflows): '
      f'{ns_total / wall * 86400:.3f} ns/day')

jax.clear_caches()


# ---------------------------------------------------------------------------
# Postprocess the trajectory
# ---------------------------------------------------------------------------
import h5py

file = h5py.File('hdf5/trajectory.hdf5', mode='r')

# Stored shape: (num_flushes, hdf5_buffer_size, num_atoms, 3)
velocities = np.array(file['velocities'])
print(velocities.shape)

# Merge the two leading axes into one list of consecutive frames,
# separated by delta_t = npt_steps * timestep_ps [ps].
velocities = velocities.reshape(-1, *file['velocities'].shape[-2:])

print(velocities.shape)  # (num_flushes * hdf5_buffer_size, num_atoms, 3)
