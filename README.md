# hamiltonian-least-squares
In this note we show how to explicitly solve Least Squares (e.g. Linear Regression) via optimal control theory/Pontryagin's Hamiltonian.

## 1. Introduction

It is well known how to "go in the *opposite direction*": discrete time Linear Quadratic Regulator (LQR) can be solved via Least Squares (LS).
(See this link for details: [[1]](#1).)
The approach requires a somewhat artificial high dimensional/formal embedding and it *completely eliminates* traces of the original system's dynamics.

In contrast, **we *directly* map a learning algorithm (like Linear Regression) onto a corresponding optimal control problem, hence treating LS as a dynamic system.** 


## 2. Motivation
2.1. We take a fresh look at the link between Artificial Neural Networks and Dynamic Systems.
- In a typical approach the connection is established via an interplay between Lagrangian multipliers (Karush-Kuhn-Tucker conditions) and the backpropagation algorithm.
Optionally, a link with Hamiltonian mechanics is used (via Pontryagin's maximum principle and the control Hamiltonian).
Then "forward" (state) and "backward" (costate) dynamics (both continuous and discrete time) are numerically simulated.

- These ideas can be traced back to this classic unparalleled book: [[2]](#2). 
Also, see these amazing articles for more recent updates: [[3]](#3), [[4]](#4)

2.2. Direct connection with a Hamiltonian system allows one to study machine learning via advanced mathematics.
- **In the sequel we will describe some of the intrinsic structures associated with such mappings**

## 3. Outline Of Work
- Linear Regression to Optimal Control *explicit/direct* mapping
- **A specific Linear Regression problem solved via new methods**

## 4. From Linear Regression To Optimal Control
We observe that Linear Regression (LR) in the most basic form, using least squares, can be understood dynamically:
we are simply building a straight line

$\large\ y = m \cdot x + b,$

whose slope *m* and y-intercept *b* are adjusted to minimize a sum of squared errors between predicted and actual (measured) y values.
The dynamic nature becomes even more apparent if we treat *x* as a time coordinate (continuous or discrete) and y as a dynamic state variable. 

We also rename the slope *m* to *u* (along with other variables) to follow standard optimal control conventions. Hence we get:

$\large\ x(t) = x(0) + u \cdot t.$

We can think of it as a solution to the following dynamic equation:

$\large\frac{dx}{dt} = u$

with the initial condition: $`\large\ x(0) = b.`$

We can also discretize it (e.g. with a 1-second time step) to obtain (think an Euler like forward approximation with total steps = time interval / step):

$`\large\ x(k+1) = x(k) + u(k)`$, where $`\large\ u(k)= u(k+1) = const.`$

Since *u* is constant, it can also be written as:

$\large\ x(k) = x(0) + u \cdot k.$


An optimization problem can be thought of as finding an optimal control *u* (the line's slope) such that it minimizes the sum of squared errors between predicted and measured states. In other words we want to minimize the following cost functional (which is a *Bolza* problem, with the *fixed time/free final point* condition):

$\large\ J = running\ cost + terminal\ cost = $

$\large\frac{1}{2}\int_{0}^{T}(x(t) - z(t))^2 \mathrm{d}t +\frac{1}{2}g\cdot(x(T) - z(T))^2$

where *z(t)* can be approximated, say, by a spline function (spline interpolation) passing through a set of N data points (coordinate pairs of *(t, z(t)).* And *x(t)* is satisfying the dynamic constraint:

$\large\dot{x} = u$

Note that in discrete case, nowhere in our calculations we will ever need the explicit formula for *z(t).* We are only interested in z's values at the N data points:
*z(1), z(2), z(3), ..., z(N).* We also use what's called a *"soft"* terminal constraint with an adjustable weight *g.* 

**Unlike the [Linear Quadratic] Tracking Control problem (or similar trajectory optimization tasks) we do not follow *z(t)* (aka "reference trajectory"). Instead our system always stays on a straight line, and we only control the slope.**

We form a Hamiltonian (also known as Pontryagin's Hamiltonian) which is typically the sum of the running cost and the costate times dynamics:

$\large\ H(x, u, \lambda, t) = \frac{1}{2}(x(t) - z(t))^2 + \lambda(t) \cdot u$

Since (in continuous case):

$\large\ x(t) = x(0) + u\cdot t$

the expression for Pontryagin's Hamiltonian becomes:

$\large\ H(u, \lambda, t) = \frac{1}{2}(x(0) + u \cdot t - z(t))^2 + \lambda(t) \cdot u$

Then for the costate equation we obtain:

$\large\frac{d \lambda}{dt} = - \frac{\partial H}{\partial x} = 0$

<br />

Therefore, the costate is constant and we can find its value using the terminal cost condition:

$\large\lambda(t) = \lambda(T)  = \frac{\partial (\frac{1}{2}g\cdot(x(T) - z(T))^2)}{\partial x} = g \cdot (x(T) - z(T))$

<br />

Minimization of the Hamiltonian with respect to *u* yields:

$\large\frac{\partial H}{\partial u} = (x(0) + u \cdot t - z(t)) \cdot t + \lambda (t) = $

$\large (x(0) + u \cdot t - z(t)) \cdot t + g \cdot (x(T) - z(T)) = 0$

<br />

After discretizing the cost functional and the dynamics we obtain the following optimization problem:

*minimize*

$\large\ J = \frac{1}{2}\sum_{k=1}^{T-1}(x(k) - z(k))^2 + \frac{1}{2} g\cdot(x(T) - z(T))^2$

*subject to*

$\large\ x(k + 1) = x(k) + u(k);\ x(0) = b.$

<br />

## 5. A Linear Regression Problem Solved Via Optimal Control

**5.1 Standard Least Squares.**

Let's consider a simple linear regression problem and let's first solve it using traditional linear least squares (LLS) and then using our optimal control based approach.
We have three (x, y) data points: *(1, 1), (2, 3), (3, 2).* We look for a line **y = m * x + b** that fits the data the best by minimizing the sum of squared residuals:

$\large\ S(m, b) = r _1 ^2 + r _2 ^2 + r _3 ^2 = $

$\large\ (m \cdot 1 + b - 1)^2 + (m \cdot 2 + b - 3)^2 + (m \cdot 3 + b - 2)^2 $

<br />


**LLS gives us the following recipe for finding *m* and *b*:**

$\large\frac{\partial S}{\partial m} = 2 \cdot (m \cdot 1 + b - 1) \cdot 1 + 2 \cdot (m \cdot 2 + b - 3) \cdot 2 + 2 \cdot (m \cdot 3 + b - 2) \cdot 3 = 0;$

$\large\frac{\partial S}{\partial b} = 2 \cdot (m \cdot 1 + b - 1) \cdot 1 + 2 \cdot (m \cdot 2 + b - 3) \cdot 1 + 2 \cdot (m \cdot 3 + b - 2) \cdot 1 = 0.$

<br />


**Simple algebra yields the final answer:**

$\large\ m = \frac {1}{2};\ b = 1;\ y = \frac {1}{2} \cdot x + 1$

<br />


This code and its output illustrate the above solution:

```python 
import numpy as np
import matplotlib.pyplot as plt

# Sample data
x = np.array([1, 2, 3])
y = np.array([1, 3, 2])

# Construct the design matrix A for a linear model y = mx + b
A = np.vstack([x, np.ones(len(x))]).T

# Solve for the coefficients (m, b) using least squares
coefficients, residuals, rank, singular_values = np.linalg.lstsq(A, y, rcond=None)

m, b = coefficients[0], coefficients[1]

print(f"Slope (m): {m:.2f}")
print(f"Intercept (b): {b:.2f}")

y_values = m * x + b

plt.plot(x, y, "--")
plt.plot(x, y_values)
plt.ylabel("Y")
plt.xlabel(f"X/Time $t$ [sec]; y = {m:.2f}x + {b:.2f}")
plt.title("LS")
```

<img width="615" height="508" alt="image" src="https://github.com/user-attachments/assets/1095e352-e93e-4d69-9067-9e9151e9703d" /> <br /><br />




**5.2 Alternative Approach Via The Hamiltonian Of Control Theory.**

Using the expression for:

$\large\frac{\partial H}{\partial u}$

<br />

and expanding the gradient of the cost functional:

$\large\nabla_u J = \sum_{k=1}^{T-1} \frac{\partial H}{\partial u(k)} + g \cdot (x(T) - z(T)) = $

$\large\sum_{k=1}^{T-1} (x(0) + u \cdot k - z(k)) \cdot k) + g \cdot (x(T) - z(T)) = 0$

<br />

we obtain:

$\large\ (x(0) + u \cdot 1 - 1) \cdot 1+ (x(0) + u \cdot 2 - 3) \cdot 2 + g \cdot (x(0) + u \cdot 3 - 2) = 0$

<br />

and if we let *x(0) = b = 1* and *g = 3* it again yields the same final answer for the slope *u*:

$\large\ u = \frac {1}{2};\ x(t) = x(0) + \frac {1}{2} \cdot t$

<br />
<br />



## References

<a id="1">[1]</a>
Stephen Boyd, (2009) [Lecture 1. Linear quadratic regulator: Discrete-time finite horizon](https://stanford.edu/class/ee363/lectures/dlqr.pdf)

<a id="2">[2]</a>
Arthur E. Bryson, Jr., Yu-Chi Ho, (1975) Applied Optimal Control: Optimization, Estimation and Control, Chapter 2

<a id="3">[3]</a>
Emanuel Todorov, (2006) [Optimal Control Theory](https://roboti.us/lab/papers/TodorovChapter06.pdf)

<a id="4">[4]</a>
Ben Recht, (2016) [Mates of Costate](https://archives.argmin.net/2016/05/18/mates-of-costate/)
