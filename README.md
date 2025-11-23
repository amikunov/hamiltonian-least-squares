# hamiltonian-least-squares
In this note we show how to explicitly solve Least Squares (e.g. Linear Regression) via optimal control theory/Pontryagin's Hamiltonian.

## 1. Introduction

It is well known how to "go in the *opposite direction*": discrete time Linear Quadratic Regulator (LQR) can be solved via Least Squares (LS).
It requires a somewhat artificial high dimensional/formal embedding and it *completely eliminates* traces of the original system's dynamics.
See this link for details: [[1]](#1).

In contrast, **we *directly* map a learning algorithm (like Linear Regression) onto a corresponding optimal control problem, hence treating LS as a dynamic system.** 


## 2. Motivation
2.1. We take a fresh look at the link between Artificial Neural Networks and Dynamic Systems.
- In a typical approach the connection is established via an interplay between Lagrangian multipliers and the backpropagation algorithm.
Optionally, a link with Hamiltonian mechanics is used (via Pontryagin's maximum principle and Pontryagin's Hamiltonian).
Then "forward" (state) and "backward" (costate) dynamics (both continuous and discrete time) are numerically simulated.

- These ideas can be traced back to this classic unparalleled book: [[2]](#2). 
Also, see these amazing articles for more recent updates: [[3]](#3), [[4]](#4)

2.2. Direct mapping to a Hamiltonian system allows one to study machine learning via modern mathematics.
- **In the sequel we will describe some of the intrinsic structures associated with these mappings**

## 3. Outline Of Work
- Linear Regression to Optimal Control *explicit/direct* mapping
- **A specific Linear Regression problem solved via new methods**

## 4. From Linear Regression To Optimal Control
We observe that Linear Regression (LR) in the most basic form, using least squares, can be understood dynamically:
we are simply building a straight line

**y = m * x + b**,

whose slope *m* and y-intercept *b* are adjusted to minimize a sum of squared errors between predicted and actual (measured) y values.
The dynamic nature becomes even more apparent if we treat *x* as a time coordinate (continuous or discrete) and y as a dynamic state variable. 

We also rename the slope *m* to *u* (along with other variables) to follow standard optimal control conventions. Hence we get:

**x(t) = x(0) + u * t**. 

We can think of it as a solution to the following dynamic equation:

<img width="56" height="39" title = "\frac{dx}{dt} = u" alt="image" src="https://github.com/user-attachments/assets/5ef15d28-e71c-4cdb-a30c-18e0dabe4f84" /> <br />

with the initial condition: **x(0) = b**.

We can also discretize it (e.g. with 1 second time step) to obtain (think an Euler like forward approximation):

**x(k+1) = x(k) + u(k)**, where **u(k)= u(k+1) = const.**

Since *u* is constant, it can also be written as:

**x(k) = x(0) + u * k.**


An optimization problem can be thought of as finding an optimal control *u* (the line's slope) such that it minimizes the sum of squared errors between predicted and measured states. In other words we want to minimize the following cost functional (a *Bolza* problem, with the *fixed time/free final point* condition):

<img width="268" height="19" title = "J = running\,cost + terminal\,cost =" alt="image" src="https://github.com/user-attachments/assets/262855c3-612b-4c66-891d-4e26901b04bd" /> <br />

<img width="308" height="51" title = "\frac{1}{2}\int_{0}^{T}(x(t) - z(t))^2 \mathrm{d}t +\frac{1}{2}g\cdot(x(T) - z(T))^2" alt="image" src="https://github.com/user-attachments/assets/9b421efb-ba39-4383-907e-fcdb66ab0979" /> <br />

where *z(t)* can be approximated, say, by a spline function (spline interpolation) passing through a set of N data points (coordinate pairs of *(t, z(t)).* And *x(t)* is satisfying the dynamic constraint:

<img width="56" height="39" title = "\dot{x} = u" alt="image" src="https://github.com/user-attachments/assets/92c1d1a4-2ab2-43a1-87ad-e6474af19ed5" /> <br />

Note, nowhere in our calculations we will ever need the explicit formula for *z(t).* We only interested in z's values at the N data points:
*z(1), z(2), z(3), ..., z(N).* We also use what's called a *"soft"* terminal constraint with an adjustable weight *g.* 

**Unlike the [Linear Quadratic] Tracking Control problem (or similar trajectory optimization tasks) we do not follow *z(t)* (aka "reference trajectory"). Instead our system always stays on a straight line, and we only control the slope.**

We form a Hamiltonian (also known as Pontryagin's Hamiltonian) which is typically the sum of the running cost and the costate times dynamics:

<img width="293" height="40" title = "\ H(x, u, \lambda, t) = \frac{1}{2}(x(t) - z(t))^2 + \lambda(t) \cdot u" alt="image" src="https://github.com/user-attachments/assets/2ec78d19-5ae5-4b7d-866e-8163d6866538" /> <br />

Since (in continuous case):

<img width="155" height="20" title = "x(t) = x(0) + u\cdot t" alt="image" src="https://github.com/user-attachments/assets/910b8560-f47e-4154-aca3-201aeb67c74e" /> <br />

the expression for Pontryagin's Hamiltonian becomes:

<img width="351" height="41" title ="\ H(u, \lambda, t) = \frac{1}{2}(x(0) + u \cdot t - z(t))^2 + \lambda(t) \cdot u" alt="image" src="https://github.com/user-attachments/assets/31ec249a-ae79-4923-9c51-ab15a9d0fb6e" /> <br />

Then for the costate equation we obtain:

<img width="123" height="41" title = "\frac{d \lambda}{dt} = - \frac{\partial H}{\partial x} = 0" alt="image" src="https://github.com/user-attachments/assets/a837f9aa-9ed4-47f2-bd99-96072950aaeb" /> <br />

Therefore, the costate is constant and we can find its value using the terminal cost condition:

<img width="398" height="44" title = "\lambda(t) = \lambda(T)  = \frac{\partial (\frac{1}{2}g\cdot(x(T) - z(T))^2)}{\partial x} = g \cdot (x(T) - z(T))" alt="image" src="https://github.com/user-attachments/assets/8bb4a903-43fd-44c7-ace2-432fc9762ecb" /> <br />

Minimization of the Hamiltonian with respect to *u* yields:

<img width="579" height="40" title = "\frac{\partial H}{\partial u} = (x(0) + u \cdot t - z(t)) \cdot t + \lambda (t) = (x(0) + u \cdot t - z(t)) \cdot t + g \cdot (x(T) - z(T)) = 0" alt="image" src="https://github.com/user-attachments/assets/0c0e62c1-33b6-410d-a263-76137e9048d5" /> <br />


After discretizing the cost functional and the dynamics we obtain the following optimization problem:

*minimize*

<img width="323" height="60" title = "J = \frac{1}{2}\sum_{k=1}^{T-1}(x(k) - z(k))^2 + \frac{1}{2} g\cdot(x(T) - z(T))^2" alt="image" src="https://github.com/user-attachments/assets/ddb6f054-355e-4668-9e7d-337a7f502900" /> <br />

*subject to*

<img width="287" height="21" title = "x(k + 1) = x(k) + u(k); x(0) = b." alt="image" src="https://github.com/user-attachments/assets/8e0b3eef-8b5f-4f8c-b529-2896e39dc36b" /> <br />


## 5. A Linear Regression Problem Solved via Optimal Control

Let's consider a simple linear regression problem and let's first solve it using traditional linear least squares (LLS) and then using our optimal control based approach.
We have three (x, y) data points: *(1, 1), (2, 3), (3, 2).* We look for a line **y = m * x + b** that fits the data the best by minimizing the sum of squared residuals:

<img width="209" height="26" title = "S(m, b) = r _1 ^2 + r _2 ^2 + r _3 ^2 = " alt="image" src="https://github.com/user-attachments/assets/78c36dfe-bb37-465d-b6b6-4bc3352b3b5a" />


<img width="334" height="24" title = "(m \cdot 1 + b - 1)^2 + (m \cdot 2 + b - 3)^2 + (m \cdot 3 + b - 2)^2 " alt="image" src="https://github.com/user-attachments/assets/30432e65-2841-4963-884a-d83d86b233af" />  <br />


**LLS gives us the following recipe for finding *m* and *b*:**

<img width="524" height="44" title = "\frac{\partial S}{\partial m} = 2 \cdot (m \cdot 1 + b - 1) \cdot 1 + 2 \cdot (m \cdot 2 + b - 3) \cdot 2 + 2 \cdot (m \cdot 3 + b - 2) \cdot 3 = 0;" alt="image" src="https://github.com/user-attachments/assets/efe6475a-67c9-4757-a91c-16924dde9336" />


<img width="524" height="44" title = "\frac{\partial S}{\partial b} = 2 \cdot (m \cdot 1 + b - 1) \cdot 1 + 2 \cdot (m \cdot 2 + b - 3) \cdot 1 + 2 \cdot (m \cdot 3 + b - 2) \cdot 1 = 0." alt="image" src="https://github.com/user-attachments/assets/db5f7030-17e0-4822-9948-54e9a2f4e65e" /> <br />


**Simple algebra yields the final answer:**

<img width="236" height="44" title = "m = \frac {1}{2}; b = 1; y = \frac {1}{2} \cdot x + 1" alt="image" src="https://github.com/user-attachments/assets/78797cc5-0e0d-412e-8a43-cdebadd02307" /> <br />



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

print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")

y_values = m * x + b

plt.plot(x, y, "--")
plt.plot(x, y_values)
plt.ylabel("Y")
plt.xlabel(f"X/Time $t$ [sec]; y = {m}x + {b}")
plt.title("LS")
```

<img width="612" height="512" alt="image" src="https://github.com/user-attachments/assets/8db8b86d-8608-487d-95d5-b2cbbb6faac0" /> <br /><br />



**Alternative approach using optimal control.**

Using the expression for:

<img width="29" height="41" title = "\frac{\partial H}{\partial u}" alt="image" src="https://github.com/user-attachments/assets/501349cc-27a6-4177-b1b5-d92455c50fac" /> <br />

and expanding the gradient of the cost function:

<img width="600" height="58" title = "\nabla _u J = \sum_{k=1}^{T-1} \frac{\partial H}{\partial u(k)} + g \cdot (x(T) - z(T)) = \sum_{k=1}^{T-1} (x(0) + u \cdot k - z(k)) \cdot k) + g \cdot (x(T) - z(T)) = 0" alt="image" src="https://github.com/user-attachments/assets/e008a5ab-6503-43f1-842e-6d1154dcca23" /> <br />

we obtain:

<img width="485" height="22" title = "(x(0) + u \cdot 1 - 1) \cdot 1+ (x(0) + u \cdot 2 - 3) \cdot 2 + g \cdot (x(0) + u \cdot 3 - 2) = 0" alt="image" src="https://github.com/user-attachments/assets/47569752-ab84-4320-a869-20ecc5a5de39" /> <br />

and if we let *x(0) = b = 1* and *g = 3* it again yields the same final answer for the slope *u*:

<img width="197" height="48" title = "u = \frac {1}{2}; x(t) = x(0) + \frac {1}{2} \cdot t" alt="image" src="https://github.com/user-attachments/assets/730e4f64-8164-4a1d-862e-f8bc21f0f4f7" /> <br />


## References

<a id="1">[1]</a>
Stephen Boyd, (2009) [Lecture 1. Linear quadratic regulator: Discrete-time finite horizon](https://stanford.edu/class/ee363/lectures/dlqr.pdf)

<a id="2">[2]</a>
Arthur E. Bryson, Jr., Yu-Chi Ho, (1975) Applied Optimal Control: Optimization, Estimation and Control, Chapter 2

<a id="3">[3]</a>
Emanuel Todorov, (2006) [Optimal Control Theory](https://roboti.us/lab/papers/TodorovChapter06.pdf)

<a id="4">[4]</a>
Ben Recht, (2016) [Mates of Costate](https://archives.argmin.net/2016/05/18/mates-of-costate/)
