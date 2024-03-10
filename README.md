# Solving-Poisson-Equation-with-DL-pt2

This repository is a continuation of the work in [Solving-Poisson-s-Equation-through-DL-for-CFD-apllications](https://github.com/pauloacs/Solving-Poisson-s-Equation-through-DL-for-CFD-apllications).

This repository contains two new variants of the Surrogate Models developed in the above project.

## U_to_gradP

- **Inputs:** U & sdf
- **Output:** grad(p)

Here a different method is being developed to improve the generalization capacity for different Re numbers.

## deltaU_to_deltaP

- **Inputs:** [U(t) - U(t-1)] & sdf
- **Output:** [p(t) - p(t-1)]

Here a new SM is being developed to drastically improve the accuracy of the Surrogate model for a given Reynolds number.

### Usage

To use it:

1. Enter the directory:
   ```bash
   cd deltaU_to_deltaP/

2. To use it check:
    ```bash
    train_script -h
    evaluate_script -h

## Create your own Surrogate Model

To actually create you own surrogate model follow the description of each entry point and:

### 1st - **Train your SM** with the train_script entry point
### 2nd - **Evaluate your SM** with the evaluate_script entry point

**Enter folder deltaU_to_deltaP for more details**

