# Exercise 3. Comparing ORB and FeNNix: ML Frameworks and Their Interaction with OpenMM

## Goal

In this exercise, you will compare two machine-learning (ML) potentials running in OpenMM:

- **ORB-V3-OMOL**, built on the PyTorch framework
- **FeNNix-Bio1M**, built on the JAX framework

You will use measured benchmark results to investigate how **system size** and **GPU allocation** affect simulation speed and resource cost.

The two models differ in their neural-network architectures and ML frameworks, and in how their model calculations are implemented within OpenMM's PythonForce interface. The benchmark results can show which model performs better under different conditions, but they cannot isolate the contribution of each difference. Part of this exercise is learning to distinguish what the measurements demonstrate from what they merely suggest.

By the end, you should be able to:

- Calculate and interpret throughput and resource cost per nanosecond.
- Explain why the fastest GPU allocation is not always the most efficient one.
- Describe how performance changes with system size for each model.
- Propose additional measurements that would help explain the results.

## Key terms

- **MIG (Multi-Instance GPU):** A way of splitting one physical GPU into several smaller, independent GPU instances. The "g" number indicates how many of the GPU's seven compute slices an instance gets (1g = 1/7 of the GPU, 3g = 3/7, and so on).
- **Throughput:** How much simulated time you get per day of computing, measured in nanoseconds per day (ns/day). Higher is faster.
- **Overhead:** Time spent on work that is not the calculation itself, such as moving data or waiting for another component.
- **Synchronization:** One component (for example, the CPU) waiting for another (for example, the GPU) to finish before continuing.

## Background

Machine-learning potentials are developed using different numerical computing frameworks. These frameworks provide the tools needed to evaluate energies and forces, but they can also influence how efficiently a model runs during MD simulations.

### PyTorch and JAX

- **PyTorch** is a widely used framework for machine learning, particularly neural networks. ORB uses PyTorch.
- **JAX** is a numerical computing framework that supports automatic differentiation and just-in-time compilation. FeNNix uses JAX to evaluate energies and forces.

JAX uses **XLA (Accelerated Linear Algebra)** to compile and optimize numerical computations for the target hardware. This can improve performance by combining operations and optimizing their execution on GPUs.

### Connecting an ML model to OpenMM

During a simulation, OpenMM needs new forces at every step, so *how* the ML model is connected to OpenMM matters a great deal.

In OpenMM-ML 1.8, both ORB and FeNNix use **PythonForce**, which allows OpenMM to call a Python function that returns the energy and forces. This provides a flexible way to integrate models built with different ML frameworks.

However, each force evaluation involves communication between OpenMM and the Python model. Coordinates must be passed to the model, and the resulting forces must be returned to OpenMM. This can introduce overhead from Python calls, data transfers, and CPU–GPU synchronization.

Although both models use the same general integration mechanism, they differ in how their model calculations are implemented: ORB uses PyTorch, while FeNNix uses JAX and XLA for compilation and execution. These differences may affect how efficiently the models use the GPU and how well their performance scales with larger GPU allocations.

In this exercise, you will investigate how these different ML frameworks and model implementations perform within the same general OpenMM–PythonForce integration approach.

## GPU allocations and resource cost

An NVIDIA H100 GPU can be allocated whole or divided into smaller MIG instances. Each allocation is assigned a cost in **RGU (Reference GPU Units)**, a standardized measure that lets us compare different hardware configurations. 

Each allocation is assigned a cost in RGU (Reference GPU Units), a standardized measure of resource usage that allows us to compare different GPU types and configurations. RGU represents the amount of GPU resources allocated, rather than the actual GPU utilization. A full H100 is assigned 12.20 RGU; smaller MIG instances are assigned a proportional share.

| GPU allocation | Share of the GPU | RGU |
|---|---:|---:|
| 1g MIG | 1/7 | 1.74 |
| 2g MIG | 2/7 | 3.49 |
| 3g MIG | 3/7 | 5.23 |
| Full H100 | 7/7 | 12.20 |

## Two metrics

**1. Simulation throughput (ns/day):** How fast the simulation runs.

**2. Resource cost per nanosecond (RGU·day/ns):** How many allocated resources are needed to produce 1 ns of simulation.

$$
\text{Resource cost per ns} =
\frac{\text{GPU allocation (RGU)}}{\text{Throughput (ns/day)}}
$$

This metric assumes you are charged for the whole allocation you reserve, whether or not the GPU is fully used. A lower value means more efficient use of resources.

You will also use a third, very practical number: **days to simulate 100 ns**.

$$
\text{Days for 100 ns} = \frac{100}{\text{Throughput (ns/day)}}
$$

### Worked example: ORB on polyAla32

| GPU allocation | Throughput (ns/day) | Resource cost per ns | Days for 100 ns |
|---|---:|---:|---:|
| 1g MIG | 1.71 | 1.02 | 58.5 |
| 2g MIG | 3.00 | 1.16 | 33.3 |
| 3g MIG | 4.40 | 1.19 | 22.7 |
| Full H100 | 5.95 | 2.05 | 16.8 |

The full H100 is the fastest, but it costs about twice as many resources per nanosecond as 1g MIG. The 1g instance is the most efficient, but the simulation takes more than three times longer.

**The fastest allocation is not necessarily the most resource-efficient one.**

**Rule of thumb:** A larger allocation uses more resources, so to be *as efficient as* 1g MIG it must also be proportionally faster. Compared with 1g MIG:

- 2g MIG needs about 2× the throughput.
- 3g MIG needs about 3× the throughput.
- Full H100 needs about 7× the throughput (12.20 ÷ 1.74).

If the speed-up is smaller than that, the resource cost per nanosecond increases.


## Benchmark results

The following tables show measured simulation throughput for each model and system, in **ns/day**.

### ORB-V3-OMOL

| System | 1g MIG | 2g MIG | 3g MIG | Full H100 |
|---|---:|---:|---:|---:|
| polyAla4 | 6.50 | 8.21 | 8.64 | 8.12 |
| polyAla8 | 4.28 | 6.11 | 6.75 | 8.39 |
| polyAla16 | 2.98 | 4.53 | 5.98 | 7.53 |
| polyAla32 | 1.71 | 3.00 | 4.40 | 5.95 |
| polyAla64 | 0.95 | 1.77 | 2.98 | 4.32 |

### FeNNix-Bio1M

| System | 1g MIG | 2g MIG | 3g MIG | Full H100 |
|---|---:|---:|---:|---:|
| polyAla4 | 12.54 | 12.83 | 13.41 | 12.94 |
| polyAla8 | 6.56 | 7.19 | 6.93 | 8.17 |
| polyAla16 | 3.21 | 3.52 | 3.40 | 4.05 |
| polyAla32 | 1.02 | 1.13 | 1.31 | 1.44 |
| polyAla64 | 0.36 | 0.36 | 0.35 | 0.33 |

**Interpreting small differences:** Differences of roughly 10% or less may be within run-to-run variability and should be interpreted cautiously. For example, 0.36 and 0.35 ns/day should be treated as effectively equivalent unless repeated measurements show otherwise.

## Tasks

Use a spreadsheet or a short script for the calculations. You need 40 resource-cost values in total: 2 models × 5 systems × 4 allocations. Enter the throughputs once and use a formula for the resource cost so you can copy it across the table.

### 1. Calculate resource cost per nanosecond

For each model, system size, and GPU allocation, calculate the resource cost per ns. 

Example: FeNNix on polyAla16 with 2g MIG:

$$
3.49 \div 3.52 \approx 0.99
$$

Also calculate the **days to simulate 100 ns** for the allocations you identify in Task 2.

### 2. Compare resource efficiency and simulation speed

For each model and system size, identify:

- The allocation with the **lowest resource cost per ns**.
- The allocation with the **highest throughput**.
- A **good compromise**, defined here as the lowest-cost allocation among those that reach at least 75% of the best throughput.

For example, if the fastest allocation achieves 8 ns/day, the threshold is 6 ns/day. Among allocations achieving at least 6 ns/day, choose the one with the lowest resource cost per ns.

If two allocations have the same value, report both as a tie.

Then answer: for one system of your choice, is the fastest option worth its extra resource cost? Use the days-for-100-ns estimates in your argument.

### 3. Investigate the effect of system size

Compare the smallest and largest systems.

- How does throughput change as the system grows, for each model?
- Does the most resource-efficient allocation change with system size?
- Does the relative performance of ORB and FeNNix change as the system grows? If so, roughly where?

### 4. Compare ORB and FeNNix

- Which model gives higher throughput for each system and allocation?
- Which model has the lower resource cost per ns? (Think about how cost and throughput are related at a fixed allocation.)
- Calculate the speed-up from 1g MIG to full H100 by dividing the full-H100 throughput by the 1g throughput. Compare this with the 7× increase in RGU. Does the larger allocation provide a proportional increase in simulation speed?
- Find cases where a larger allocation gives little or no extra speed. Which are they?

### 5. Discuss the results

Based on your calculations, discuss:
- Why might the largest GPU allocation not be the most resource-efficient?
- What could limit the performance of an ML potential as the system gets larger?
- How might the ML framework and OpenMM integration approach affect performance?
- What additional measurements could help explain the performance differences?
Consider measurements you could collect while a simulation is running:


| Measurement | What it could help reveal |
|---|---|
| GPU utilization | Whether the GPU is kept busy or frequently idle |
| GPU memory usage | How much GPU memory the model requires |
| CPU utilization | Whether CPU-side work may limit performance |


For example, if GPU utilization is low, could the simulation be limited by something other than GPU computing capacity?

Important: Throughput measurements can show that performance changes, but not necessarily why. Treat explanations as hypotheses unless additional measurements support them.


## Key takeaway

Choosing a GPU allocation means balancing **simulation speed** against **resource efficiency**. A larger GPU may finish a simulation sooner but use substantially more resources per nanosecond. Splitting a GPU with MIG also lets several independent simulations run side by side, which is one reason efficiency matters on a shared cluster.

The best choice depends on whether turnaround time or efficient use of shared computing resources is the priority.------------





sbatch --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 submit_FENNIX.sh
sbatch --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 submit_FENNIX.sh
sbatch --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 submit_FENNIX.sh
sbatch --gpus=h100:1 submit_FENNIX.sh
sbatch --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 submit_ORB-OMOL.sh
sbatch --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 submit_ORB-OMOL.sh
sbatch --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 submit_ORB-OMOL.sh
sbatch --gpus=h100:1 submit_ORB-OMOL.sh

