# CMSA_assignment
This is a project for assignment of course "An Introduction to Computational Neuroscience: from Brain Simulation to NeuroAI"
The goal is to replicate the work of Chen, Guozhang, and Pulin Gong (2019) https://www.nature.com/articles/s41467-019-12918-8.

## 1. Overview
This project implements a **CMSA (Computing by modulating spontaneous cortical activity)** using **PyTorch**. It simulates a Spiking Neural Network (SNN) with **LIF (Leaky Integrate-and-Fire)** neurons to explore spatiotemporal patterns like "bump-wave" transitions.

## 2. Core Features
* **LIF Neuron Layer**: Features membrane potential integration and refractory periods.
* **Synaptic Dynamics**: Exponentially decaying synaptic currents (AMPA).
* **Spatial Connectivity**: Gaussian kernel-based weights with periodic boundary conditions.
* **Performance**: GPU acceleration via CUDA, you can simlate a 300*300 network on personal PC(8 GB video-memory).

## 3. Key Visualizations

### 3.1 Pattern Transition
Transitions between static activity (Bump) and moving activity (Wave) are controlled by synaptic weights.

![BumpandWave.png]

### 3.2 Phase transition
When the connection parameters of the model are changed, the order parameter undergoes a phase transition.

![Phase_trans.png]

## 4. How to Run
1. Install dependencies: `pip install torch matplotlib numpy`.
2. Open `CMSA_Solution.ipynb`.
3. Run all cells to initialize the `Network` class and start the simulation.
