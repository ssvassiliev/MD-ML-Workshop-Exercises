# Exercise 4 — Scaling Up with SO3LR: Full-System ML Molecular Dynamics with JAX-MD

In the previous exercises, we explored hybrid ML/MM simulations, in this exercise, we take a different approach: **the entire system is described by an ML potential.**

We will use SO3LR, a machine-learning interatomic potential designed to combine local atomic environments with explicit longer-range physical interactions. Unlike the previous exercises, SO3LR does not run through OpenMM. Instead, it uses JAX-MD, a molecular dynamics engine built on JAX. This exercise demonstrates how we can explore large-scale full-system ML-driven molecular dynamics.

### Goal

 Learn how to use SO3LR with JAX-MD to simulate a larger molecular system.

By the end of this exercise, you should be able to:

- Understand the roles of SO3LR, JAX, and JAX-MD.
- Prepare an AMBER-generated structure for an SO3LR simulation.
- Run a short molecular dynamics simulation with JAX-MD..
- Restart a simulation and convert its trajectory for further analysis.

## Introduction: From hybrid ML/MM to full-system ML

In the previous exercises, we used OpenMM-ML to combine classical molecular mechanics force field with an ML potential. This hybrid approach allows us to apply computationally expensive ML potentials only where they are most useful.

In this exercise we will  use different approach. Instead of dividing the system into ML and MM regions, we will evaluate the entire system using a single ML potential.

This approach removes the need to define an ML region and an MM region. However, it also means that the ML potential must describe all relevant interactions in the system, including those involving solvent molecules.

## 1. From an AMBER structure to an ML simulation

We will start with a molecular system prepared and equilibrated  using the familiar AMBER workflow. Before running the simulation, the structure must be converted into the format expected by SO3LR and JAX-MD.

The preparation involves several steps:

- Read atomic coordinates, elements, and periodic box information from the AMBER topology and  restart.
- Remove virtual sites (if any are present, for example if you used OPC water model), since SO3LR expects real atoms.
- Convert the atomic coordinates and periodic box to JAX-MD's expected format, using metal units (Å, eV, ps, and amu).
- Set up the SO3LR potential and its neighbor lists.

Unlike the previous exercises, we do not assign different force fields to different parts of the system. All real atoms are evaluated by SO3LR.

## 2. How SO3LR works with JAX-MD

SO3LR does not use OpenMM as its molecular dynamics engine. Instead, it relies on three components:

- SO3LR - the ML potential
- JAX - a Python framework for numerical computing.  It supports automatic differentiation and compilation of numerical operations for efficient execution on GPUs and CPUs.One particularly important feature is JIT (just-in-time) compilation. JAX can compile computational operations into optimized code for the target hardware. 
- JAX-MD -  molecular dynamics engine, provides the algorithms needed to perform molecular dynamics, including integration and temperature and pressure control.

The relationship is straightforward:

- SO3LR calculates energies and forces.
- JAX provides the computational framework and acceleration.
- JAX-MD uses the forces to propagate the system through time.


### A Note About JAX Compilation

One important difference from our previous OpenMM simulations is that JAX uses **just-in-time (JIT) compilation**. Before executing operations efficiently, JAX traces, optimizes, and compiles the computational functions. This introduces additional overhead whenever a function is encountered for the first time with a new input signature (i.e., different array shapes or data types).

Consequently:

- **The first execution can be noticeably slower.** Subsequent executions are typically much faster because JAX reuses the compiled functions, provided the input shapes and data types remain unchanged.
- **Neighbor-list overflows can trigger recompilation.** When a neighbor list exceeds its allocated capacity, it must be reallocated with a larger capacity. This changes the array shapes, forcing JAX to recompile the affected functions. The simulation cycle in which this happens will therefore be slower. Overflows are generally rare after the initial cycles because the allocation includes additional capacity controlled by `capacity_multiplier`.

When benchmarking performance, it is important to distinguish compilation overhead from actual MD execution time. **Exclude the initial compilation cycle and any cycles involving neighbor-list reallocation from performance measurements.**  

This distinction is particularly important for short simulations, where compilation overhead can account for a substantial fraction of the total runtime.

## 3. Running molecular dynamics

Once the system has been prepared and the SO3LR potential initialized, we can run molecular dynamics.

The simulation uses familiar MD parameters:

- **Timestep:** the integration time step.
- **Number of steps:** how long the simulation runs.
- **Temperature:** the target simulation temperature.
- **Pressure:** the target pressure.
- **Ensemble:** the statistical ensemble used for the simulation.

For this exercise, we use an NPT ensemble. Temperature and pressure are controlled using a Nose–Hoover thermostat and barostat.


### Minimization before MD

An additional practical consideration is the initial structure. An AMBER-prepared structure is not necessarily an ideal starting configuration for an ML potential. The initial geometry may be inconsistent with the potential's preferred equilibrium geometry, particularly for solvent molecules.

For example,  a system containing TIP3P water can become unstable when SO3LR MD is started directly from the AMBER restart, while an OPC water system can run without this additional preparation. For this reason, we provide a minimization script. The minimization is intended to relax unfavorable initial configurations and reduce the risk of instability at the beginning of the simulation.


## 4. Practical work

We provide several scripts to explore the SO3LR/JAX-MD workflow.

### 1. Run MD from an AMBER restart

Start with the AMBER restart file and run a short molecular dynamics simulation using SO3LR.

Your task:

1. Inspect the input script `jaxmd_input_so3lr_amber_restart.py`and identify the main simulation parameters.
2. Identify where the SO3LR potential is initialized.
3. Locate the thermostat, barostat, and integration settings.
4. Submit the simulation with `submit_so3lr.sh`.
5. Examine the resulting HDF5 trajectory.

### 2. Run Minimization Followed by MD

Before starting MD, perform **20 cycles of FIRE energy minimization** on the initial AMBER system using `jaxmd_input_so3lr_minimize_amber_restart.py`.

The minimization will save the resulting structure to `hdf5/minimized.hdf5`. Use this minimized structure as the starting point for a **20 ps MD simulation**, using `jaxmd_input_so3lr_hdf5_restart.py`.

Compare the behavior with the previous run, which started directly from the initial AMBER structure.

Consider:

- Does the simulation start more smoothly?
- Are there differences in the initial energy and forces?
- Why might minimization be particularly important when transferring a structure between different potentials?

### 3. Restart an Existing Simulation

MD simulations are often much longer than a single job allocation allows. Therefore, the ability to restart a simulation from a previously saved state is essential.

We provide a script, `jaxmd_input_so3lr_hdf5_restart.py`, that reads an existing SO3LR HDF5 trajectory and continues the simulation from its last saved configuration.

Your task:

1. Inspect how the last configuration is retrieved from the HDF5 file.
2. Restart the simulation from the existing minimized system or trajectory.
3. Verify that the new trajectory continues from the final configuration of the previous run.


### 4. Convert the trajectory for analysis

SO3LR writes simulation trajectories in HDF5 format.

For compatibility with familiar molecular visualization and analysis tools, we also provide a script to convert the HDF5 trajectory into AMBER NetCDF format.

Your task:

1. Convert the HDF5 trajectory to AMBER NetCDF.
2. Load the trajectory into a molecular visualization program.
3. Examine the resulting molecular motion.

This step illustrates that although the simulation engine and trajectory format differ from our previous OpenMM exercises, we can still use familiar tools for downstream analysis.

---

## Key takeaways

In this exercise, we moved from hybrid ML/MM simulations to full-system ML molecular dynamics.

The main differences are:

- Instead of applying an ML potential to only a selected region, SO3LR describes the entire system.
- AMBER provides the initial structure, but its force field is not used during the simulation.
- SO3LR supplies the energy and forces, while JAX and JAX-MD provide the computational framework and MD algorithms.
- JAX JIT compilation introduces an initial overhead that should be considered when measuring performance.
- Initial structure preparation and minimization can be important when switching between different potentials.

**The broader idea:** ML potentials are not limited to replacing the force field for a small region of a molecular system. With suitable models and computational approaches, they can also be used to drive molecular dynamics for an entire system.

