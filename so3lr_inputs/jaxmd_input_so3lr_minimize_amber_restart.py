#!/usr/bin/env python
"""FIRE energy minimization of an AMBER-prepared system with SO3LR + jax_md.
Saves the minimized structure to HDF5."""
import jax
import jax.numpy as jnp
import jax_md
import numpy as np
import pathlib
import parmed as pmd
import h5py

from ase import Atoms
from ase.geometry import cellpar_to_cell

from so3lr import to_jax_md
from so3lr import So3lrPotential

# For float64 simulations, call this before any JAX computation:
# jax.config.update('jax_enable_x64', True)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
prmtop = 'system.parm7'
restrt = 'system.rst7'
output_file = 'hdf5/minimized.hdf5'

min_cycles = 5        # Maximum number of Python-level cycles
min_steps = 10        # FIRE steps per cycle (one compiled fori_loop)
fmax_tol = None       # Stop early once max |F| [eV/A] falls below this; None = always run min_cycles

# ---------------------------------------------------------------------------
# System setup: read the AMBER topology + restart
# ---------------------------------------------------------------------------
struct = pmd.load_file(prmtop, restrt)

# Drop virtual sites (e.g. OPC "EPW", atomic number 0): SO3LR only knows real atoms.
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

species = jnp.array(atoms.get_atomic_numbers())
num_atoms = len(species)

# The box must be a 1-element array (cubic cell, edge length in A).
box = jnp.array([np.array(atoms.get_cell())[0, 0]])

fractional_coordinates = True
displacement, shift = jax_md.space.periodic_general(
    box=box, fractional_coordinates=fractional_coordinates
)

positions_init = jnp.array(atoms.get_positions())
if fractional_coordinates:
    positions_init = positions_init / box

# ---------------------------------------------------------------------------
# SO3LR energy function and neighbor lists
# ---------------------------------------------------------------------------
neighbor_fn, neighbor_fn_lr, energy_fn = to_jax_md(
    potential=So3lrPotential(
        dtype=jnp.float64  # or float32 for single precision
    ),
    displacement_or_metric=displacement,
    box_size=box,
    species=species,
    capacity_multiplier=1.25,
    buffer_size_multiplier_sr=1.25,
    buffer_size_multiplier_lr=1.25,
    minimum_cell_size_multiplier_sr=1.0,
    disable_cell_list=False,
    fractional_coordinates=fractional_coordinates
)

energy_fn = jax.jit(energy_fn)
force_fn = jax.jit(jax_md.quantity.force(energy_fn))

nbrs = neighbor_fn.allocate(positions_init, box=box)
nbrs_lr = neighbor_fn_lr.allocate(positions_init, box=box)

# ---------------------------------------------------------------------------
# Energy minimization (FIRE)
# ---------------------------------------------------------------------------
fire_init, fire_apply = jax_md.minimize.fire_descent(
    energy_fn,
    shift,
    dt_start=0.05,
    dt_max=0.1,
    n_min=2
)

fire_apply = jax.jit(fire_apply)
fire_state = fire_init(
    positions_init,
    box=box,
    neighbor=nbrs.idx,
    neighbor_lr=nbrs_lr.idx
)


@jax.jit
def step_fire_fn(i, carry):
    """One FIRE step, then refresh both neighbor lists."""
    fire_state, nbrs, nbrs_lr = carry

    fire_state = fire_apply(
        fire_state,
        neighbor=nbrs.idx,
        neighbor_lr=nbrs_lr.idx,
        box=box
    )

    nbrs = nbrs.update(fire_state.position, neighbor=nbrs.idx, box=box)
    nbrs_lr = nbrs_lr.update(fire_state.position, neighbor_lr=nbrs_lr.idx, box=box)

    return fire_state, nbrs, nbrs_lr


print('Cycle\tE [eV]\tFmax [eV/A]')
print('--------------------------')

E, Fmax = None, None
cycle = 0
while cycle < min_cycles:
    new_fire_state, new_nbrs, new_nbrs_lr = jax.block_until_ready(
        jax.lax.fori_loop(
            0,
            min_steps,
            step_fire_fn,
            (fire_state, nbrs, nbrs_lr)
        )
    )

    # If a neighbor list overflowed, the block's result is invalid: reallocate
    # and redo the cycle from the previous state.
    if new_nbrs.did_buffer_overflow or new_nbrs_lr.did_buffer_overflow:
        print('Neighbor list overflowed, reallocating.')
        nbrs = neighbor_fn.allocate(fire_state.position, box=box)
        nbrs_lr = neighbor_fn_lr.allocate(fire_state.position, box=box)
        continue

    fire_state, nbrs, nbrs_lr = new_fire_state, new_nbrs, new_nbrs_lr

    E = float(energy_fn(
        fire_state.position,
        neighbor=nbrs.idx,
        neighbor_lr=nbrs_lr.idx,
        box=box
    ))
    F = force_fn(
        fire_state.position,
        neighbor=nbrs.idx,
        neighbor_lr=nbrs_lr.idx,
        box=box
    )
    Fmax = float(np.abs(F).max())

    print(f'{cycle}\t{E:.4f}\t{Fmax:.4f}')
    cycle += 1

    if fmax_tol is not None and Fmax < fmax_tol:
        print(f'Converged: Fmax < {fmax_tol} eV/A after {cycle} cycles.')
        break

# ---------------------------------------------------------------------------
# Save minimized structure (HDF5)
# ---------------------------------------------------------------------------
# Fractional -> Cartesian [A] (only correct for cubic cells).
positions_cart = np.asarray(
    jax_md.space.transform(box=box, R=fire_state.position), dtype=np.float64
)

out = pathlib.Path(output_file).expanduser().resolve()
out.parent.mkdir(parents=True, exist_ok=True)

with h5py.File(out, 'w') as f:
    # Same layout as the MD trajectory file: (n_frames, ...), here one frame.
    f.create_dataset('positions', data=positions_cart[None])                   # (1, N, 3) [A]
    f.create_dataset('velocities', data=np.zeros((1, num_atoms, 3)))           # (1, N, 3) [A/ps]
    f.create_dataset('box', data=np.asarray(box, dtype=np.float64)[None])      # (1, 1) [A]
    f.create_dataset('species', data=np.asarray(species))                      # (N,)
    f.attrs['energy_eV'] = E
    f.attrs['fmax_eV_per_A'] = Fmax
    f.attrs['cycles_run'] = cycle

print(f'Saved minimized structure to {out}')
jax.clear_caches()
