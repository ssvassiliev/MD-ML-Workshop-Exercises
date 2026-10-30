# Exercise 1. From MM to ML/MM with ANI-2x

## Overview

In this exercise, you will start with a working OpenMM simulation of a fully solvated polyalanine peptide using a conventional molecular mechanics (MM) force field. You will modify the input script to replace the MM treatment of the peptide with the ANI-2x machine-learning potential, while retaining the conventional MM treatment of the surrounding water molecules.

### How does ML/MM work?

In this exercise, we use a hybrid ML/MM approach, where different parts of the molecular system are described using different potential models.

- The polyalanine peptide is treated using the ANI-2x machine-learning potential.
- The surrounding water molecules are treated using a conventional MM force field.

The interactions between the peptide and solvent are also described using MM. This approach is known as *mechanical embedding*.

The total potential energy consists of three contributions:

$$
E_{\mathrm{total}} =
E_{\mathrm{ML}}^{\mathrm{peptide}} +
E_{\mathrm{MM}}^{\mathrm{solvent}} +
E_{\mathrm{MM}}^{\mathrm{peptide-solvent}}
$$


An important limitation is that the ML potential does not explicitly account for the electrostatic polarization of the peptide by its environment.

Keep this in mind when interpreting the results: the peptide is described by a more sophisticated potential, but its interactions with the surrounding solvent remain classical.

## Tasks

### 1. Understand the existing MM simulation
- Examine the OpenMM input script `openmm_input_MM.py`.
- Identify where the force field is loaded and the OpenMM system is created.
- Run a short MM-only simulation using submission script `submit_MM.sh`.

### 2. Introduce the ML potential
- Create `openmm_input_ANI2X.py` and `submit_ANI2X.sh` by adapting the `openmm_input_MM.py` and `submit_MM.sh` templates.
- Load the ANI-2x potential using OpenMM-ML.
- Select the atoms belonging to the polyalanine peptide.
- Use `createMixedSystem()` to assign ANI-2x to the peptide while retaining MM for the solvent.

### 3. Run the hybrid ML/MM simulation
- Keep the same initial structure and simulation parameters.
- Run a short MD trajectory.
- Check that the simulation completes and produces reasonable output.

### 4. Verify and benchmark the hybrid simulation
- Confirm that the simulation completes successfully.
- Check the energy and temperature output for obvious anomalies.
- Compare the computational performance of MM-only and ML/MM simulations.
- Consider the additional computational cost of using an ML potential.

## Learning objectives

By completing this exercise, you will learn:
- How an OpenMM simulation is constructed.
- How ML potentials are integrated into an existing MD engine.
- The difference between conventional MM and hybrid ML/MM simulations.
- How to assign different potential models to different parts of a molecular system.

## Questions for discussion
1. Which parts of the molecular system are treated using MM and which using ANI-2x?
2. What changes are needed in the OpenMM input script to introduce an ML potential?
3. How does the computational performance compare between MM and ML/MM?
