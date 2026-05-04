# Monte Carlo Simulation of the 2D Ising Model

**Author:** Hope Kwadzo Dzamesi  
**Course:** Kinetic Theory and Stochastic Simulations  
**Institution:** University of L'Aquila, Department of Information Engineering and Mathematics  
**Supervisor:** Prof. Matteo Colangeli  
**Tools:** MATLAB

## Overview

This project implements a Metropolis-Hastings Monte Carlo simulation of the 
two-dimensional Ising model — a fundamental model of spin interactions on a 
square lattice used to study phase transitions in statistical physics.

The simulation investigates how a system of spins transitions between ordered 
(ferromagnetic) and disordered (paramagnetic) states as temperature crosses 
the critical threshold Tc ≈ 2.269.

## Key Results

- Phase transition clearly observed at Tc ≈ 2.269
- Magnetization drops sharply from ~1 to ~0 at the critical temperature
- Specific heat and magnetic susceptibility peak near Tc
- Microscopic configurations visualised at T=2.0 (ordered) and T=2.5 (disordered)

## Files

- `Initialization.m` — Initializes spin lattice (all +1, all -1, or random)
- `Neighbor.m` — Computes nearest neighbours with periodic boundary conditions
- `Metropolis.m` — Core Metropolis-Hastings algorithm for spin updates
- `Monte_Carlo_Project.m` — Main script: runs full simulation across temperatures
- `Monte.m` — Streamlined alternative implementation
- `mc.m` — Analytical formulas and additional simulation variants
- `results/` — Output plots: magnetization, energy, specific heat, susceptibility

## Connection to Statistical Learning

The Metropolis-Hastings algorithm implemented here is the same algorithmic 
foundation used in Markov Chain Monte Carlo (MCMC) methods in Bayesian 
inference and probabilistic machine learning. The concept of sampling from 
a probability distribution to estimate expectations — central to this 
simulation — directly underpins modern approaches to uncertainty 
quantification in statistical models.

## How to Run

Open MATLAB and run:
```matlab
Monte_Carlo_Project.m
```
Enter lattice size (recommended: L=100) and initial condition when prompted:
- 1 = All spins positive
- 2 = All spins negative  
- 3 = Random spin configuration
