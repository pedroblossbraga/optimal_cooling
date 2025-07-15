# Optimal Cooling

We aim to implement a solution to the Dual SDP for optimal heat transfer, which initially uses the average temperature gradient as a measure of transport efficiency.

## Execution
To run, open matlab, and execute: 

- Newton method + Gradient descent (primal) / ascent (dual): 
    
    newton_ocp.m

- Baseline: interior point method

    ocp_main.m

## Requirements

Make sure to install MOSEK and YALMIP, required for the baseline model. 