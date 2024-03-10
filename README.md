# Solving-Poisson-Equation-with-DL-pt2

This repo is a continuation of the work in https://github.com/pauloacs/Solving-Poisson-s-Equation-through-DL-for-CFD-apllications. 


This repository contains two new variants of the Surrogate Models developed in the above project.

## U_to_gradP - Inputs: U & sdf -> Output: grad(p)

Here a different method is being developed to improve the generalization capacity for different Re numbers.


## deltaU_to_deltaP - Inputs: [U(t) - U(t-1)] & sdf -> Output: [p(t) - p(t-1)]

Here a new SM is being developed to drastically improve the accuracy of the Surrogate model for a given Reynolds number.

**Enter folder deltaU_to_deltaP for more details**

