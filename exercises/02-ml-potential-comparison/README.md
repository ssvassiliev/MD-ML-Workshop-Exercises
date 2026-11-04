# Exercise 2. Comparing ML Potentials: ANI-2x, AIMNet2, and MACE-OFF24

## Goal
In this exercise, you will run MD simulations using three different ML potentials: ANI-2x, AIMNet2, and MACE-OFF24.

Using fully solvated polyalanine peptides of different lengths, you will investigate how the choice of ML potential and system size affect simulation performance and GPU resource utilization.

By the end of this exercise, you should be able to:
- Load and use different pretrained ML potentials with OpenMM-ML.
- Run ML/MM simulations using different potentials.
- Compare simulation throughput and GPU resource utilization.
- Understand how model choice and system size affect computational performance.
- Examine basic MD properties, including potential energy and temperature.

## Practical work
### 1. Set up the models
You will compare three pretrained ML potentials:

- ANI-2x
- AIMNet2
- MACE-OFF24

Create separate OpenMM input files for AIMNet2 and  MACE-OFF24 models, using your  input file from Exercise 1 as a template. 

Modify the input files to load the corresponding ML potential ('aimnet2' and 'mace-off24-medium'). Keep the starting structure and simulation settings consistent across all three models.

### 2. Run the simulations. 
#### Systems. 

The test systems are fully solvated polyalanine peptides of different lengths. 

| System | ML atoms | Total atoms |
|---|---:|---:|
| polyAla4 | 52 | 5,095 |
| polyAla8 | 96 | 8,069 |
| polyAla16 | 184 | 17,530 |
| polyAla32 | 360 | 52,694 |
| polyAla64 | 712 | 222,607 | 
   

#### Simulation parameters

Run 20 ps of Langevin MD for each model and system size. Use the same starting structure and simulation parameters1. Only the ML potential should change between simulations of the same system. 

#### Preparing and running the jobs.

During the workshop, you do not need to run every model on every system size. The main goal is to prepare the input files and verify that the simulations run successfully.  

1. Create separate OpenMM input files for ANI-2x, AIMNet2, and MACE-OFF24.  
3. Prepare the corresponding Slurm submission scripts. 
4. Choose a representative system and submit a short test job for each model. 
4. Check the output to confirm that the simulations complete successfully and report performance in ns/day. 

#### Switching between input structures

The Python input files expect the AMBER topology and coordinate files to have fixed names:

- `system.parm7`
- `system.rst7`

Rather than editing the Python scripts or copying input files for every system, use symbolic links to point these filenames to the desired system.

For example, to use `polyAla16`:

```bash
ln -sf ../amber_systems/polyala16/system.parm7 system.parm7
ln -sf ../amber_systems/polyala16/system.rst7 system.rst7
```

To switch to another system, simply update the links:

```bash
ln -sf ../amber_systems/polyala64/system.parm7 system.parm7
ln -sf ../amber_systems/polyala64/system.rst7 system.rst7
```

The Python input files do not need to be changed.

**Important:** Make sure the symbolic links point to the intended system before submitting a job. Since the input filenames are the same, the simulation will use whichever files the links currently reference.  

## 3. Investigate GPU performance
The goal is to understand how different ML potentials use GPU resources and how this relates to actual simulation speed.

For each model and system size, compare:

1. **GPU utilization:** Is the GPU being used during the simulation?
2. **GPU saturation:** Is the simulation making effective use of the allocated GPU? Use the reference tables below to assess whether utilization increases with system size.
3. **Simulation throughput:** Which model is fastest? Compare throughput in ns/day.
4. **System size:** How does increasing the peptide length affect performance? Does the relative performance of the models change as the system grows?
5. **GPU utilization versus throughput:** Does higher GPU utilization necessarily mean higher simulation throughput? Compare GPU utilization, memory usage, and ns/day. Can you explain any differences?

On Nibi, GPU job statistics are available through the [Nibi portal](https://portal.nibi.sharcnet.ca).

**Note:** GPU utilization and memory metrics are available only for jobs using an entire GPU. These metrics are not available for MIG allocations.

### Reference performance: polyAla4

| Model | GPU utilization | GPU memory | Throughput |
|---|---:|---:|---:|
| ANI-2x | 58% | 2.1 GB | 6.97 ns/day |
| AIMNet2 | 28% | 2.1 GB | 6.77 ns/day |
| MACE-OFF24 | 49% | 2.6 GB | 6.67 ns/day |

### Reference performance: polyAla16

| Model | GPU utilization | GPU memory | Throughput |
|---|---:|---:|---:|
| ANI-2x | 53% | 2.2 GB | 6.39 ns/day |
| AIMNet2 | 38% | 2.1 GB | 6.82 ns/day |
| MACE-OFF24 | 82% | 5.0 GB | 5.13 ns/day |

### Reference performance: polyAla64

| Model | GPU utilization | GPU memory | Throughput |
|---|---:|---:|---:|
| ANI-2x | 53% | 2.2 GB | 5.58 ns/day |
| AIMNet2 | 38% | 2.4 GB | 4.92 ns/day |
| MACE-OFF24 | 92% | 27.2 GB | 1.77 ns/day |

### 3. Discussion

Consider the following questions:
- Which model makes the most effective use of the GPU?
- Does GPU utilization increase with the number of ML atoms?
- Why might MACE-OFF24 have higher GPU utilization but lower throughput than ANI-2x?
- How does GPU memory consumption change with model choice and system size?
- Which model would you choose if your priority were simulation speed? Would your choice change for a much larger system?

Remember that GPU utilization is not the same as computational efficiency. A GPU can be highly utilized while performing fewer simulation steps per second than another model.

## 4. Inspect reference simulation output
The simulations in this exercise are relatively short and are primarily intended for performance benchmarking.

To illustrate the behavior of longer MD simulations, you are also provided with reference output from nanosecond-scale simulations.

Examine the supplied energy, temperature, and volume data and consider the following questions.

### Potential energy

1. How does the potential energy evolve during the simulation?
2. Does it fluctuate around a relatively stable range, or is there evidence of systematic drift?

### Temperature

3. Does the temperature fluctuate around the target value?
4. What fluctuations would you expect from Langevin dynamics?

### Volume

5. Does the simulation volume remain constant or fluctuate?
6. Why do you expect volume fluctuations in an NPT simulation?

**Note:** These are basic checks of simulation behavior and stability. They do not establish the accuracy of an ML potential. Also, absolute potential energies from different models are not necessarily directly comparable.

## 5. Summary

After completing this exercise, you should be able to explain:

- How different ML potentials can be integrated into OpenMM simulations.
- How model choice and system size affect GPU resource usage and simulation throughput.
- Why GPU utilization alone is not sufficient to compare performance.
- Why computational performance and physical accuracy are different properties of an ML potential.

## Optional extension: Working with local ML models

In research applications, you may want to use model weights stored locally rather than relying on automatically downloaded pretrained models.

This is particularly useful when working with custom or fine-tuned ML potentials.

Try the following:

1. Download a pretrained ML potential and save its weights in a local directory.
2. Modify the OpenMM input file to load the model using an explicit local path (`modelPath`).
```python
potential = MLPotential(
    "mace",
    modelPath=MODELPATH,
    device="cuda")
```
3. Run a short simulation and verify that the locally loaded model works.
4. Investigate what additional tools and data would be required to fine-tune the model for a particular molecular system.

**Discussion:** Why might a researcher want to fine-tune a pretrained ML potential rather than use it without modification?

