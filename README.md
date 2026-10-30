## MD-ML-Workshop-Exercises
Workshop materials and hands-on exercises introducing machine-learning potentials for molecular dynamics with OpenMM-ML and SO3LR.

The exercises introduce ML/MM simulations, explore different ML potentials, and investigate GPU performance and resource efficiency.

## Setup
- Clone the repo
- Download simulation systems

### Exercise 1. From MM to ML/MM with ANI-2x.
Modify the openmm input script to replace the force-field treatment of the polyalanine peptide with ANI-2x, while retaining the conventional MM treatment of the surrounding water molecules.

### Exercise 2. Compare ML potentials: ANI-2x, AIMNet2, and MACE-OFF24. 
How does performance vary with the model and system size?

### Exercise 3. Compare integration approaches: ORB OMol and FeNNix-Bio1.
How do the ML framework and the OpenMM integration layer affect GPU utilization and throughput? Introduce PyTorch, JAX, PythonForce, and JAX-specific integration approaches here.

### Exercise 4. Scaling up with SO3LR: 
Move from an ML region embedded in a conventional MM system to simulations where the full system is treated with an ML potential, using SO3LR's own JAX-based simulation engine.