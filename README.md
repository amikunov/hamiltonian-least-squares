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

<img width="268" height="19" alt="image" title = "J = running\,cost + terminal\,cost =" src="https://github.com/user-attachments/assets/1b6b35da-5935-4736-a5ae-c3cdf3e472c2" /> <br />

<img width="308" height="51" title = "\frac{1}{2}\int_{0}^{T}(x(t) - z(t))^2 \mathrm{d}t +\frac{1}{2}g\cdot(x(T) - z(T))^2" alt="image" src="https://github.com/user-attachments/assets/2e5fd640-d9bb-48e0-bf96-945305b8c38b" /> <br />

where *z(t)* can be approximated, say, by a spline function (spline interpolation) passing through a set of N data points (coordinate pairs of *(t, z(t)).* And *x(t)* is satisfying the dynamic constraint:

<img width="56" height="39" title = "\dot{x} = u" alt="image" src="https://github.com/user-attachments/assets/4b87f49d-9e59-41c7-958d-8466cf95692b" /> <br />

Note, nowhere in our calculations we will ever need the explicit formula for *z(t).* We only interested in z's values at the N data points:
*z(1), z(2), z(3), ..., z(N).* We also use what's called a *"soft"* terminal constraint with an adjustable weight *g.* 

**Unlike the [Linear Quadratic] Tracking Control problem (or similar trajectory optimization tasks) we do not follow *z(t)* (aka "reference trajectory"). Instead our system always stays on a straight line, and we only control the slope.**

We form a Hamiltonian (also known as Pontryagin's Hamiltonian) which is typically the sum of the running cost and the costate times dynamics:

<img width="293" height="40" title = "\ H(x, u, \lambda, t) = \frac{1}{2}(x(t) - z(t))^2 + \lambda(t) \cdot u" alt="image" src="https://github.com/user-attachments/assets/b40f114c-928a-46fe-a230-226b3644fa10" /> <br />

Since (in continuous case):

<img width="155" height="20" title = "x(t) = x(0) + u\cdot t" alt="image" src="https://github.com/user-attachments/assets/3dcf5e53-4156-4b63-a041-827d8a9def1b" /> <br />

the expression for Pontryagin's Hamiltonian becomes:

<img width="351" height="41" title ="\ H(u, \lambda, t) = \frac{1}{2}(x(0) + u \cdot t - z(t))^2 + \lambda(t) \cdot u" alt="image" src="https://github.com/user-attachments/assets/fae8a633-74f6-4e77-bfed-39099528b206" /> <br />

Then for the costate equation we obtain:

<img width="123" height="41" title = "\frac{d \lambda}{dt} = - \frac{\partial H}{\partial x} = 0" alt="image" src="https://github.com/user-attachments/assets/81521010-7447-4338-8e79-5b2f08cff4e7" /> <br />


Therefore, the costate is constant and we can find its value using the terminal cost condition:

<img width="398" height="44" title = "\lambda(t) = \lambda(T)  = \frac{\partial (\frac{1}{2}g\cdot(x(T) - z(T))^2)}{\partial x} = g \cdot (x(T) - z(T))" alt="image" src="https://github.com/user-attachments/assets/4d1111a5-956b-45ce-a65a-a97040e8cf52" /> <br />


Minimization of the Hamiltonian with respect to *u* yields:

<img width="579" height="40" title = "\frac{\partial H}{\partial u} = (x(0) + u \cdot t - z(t)) \cdot t + \lambda (t) = (x(0) + u \cdot t - z(t)) \cdot t + g \cdot (x(T) - z(T)) = 0" alt="image" src="https://github.com/user-attachments/assets/49ec05cd-42fd-4d02-a48b-597f709169ea" /> <br />

After discretizing the cost functional and the dynamics we obtain the following optimization problem:

*minimize*

<img width="323" height="60" title = "J = \frac{1}{2}\sum_{k=1}^{T-1}(x(k) - z(k))^2 + \frac{1}{2} g\cdot(x(T) - z(T))^2" alt="image" src="https://github.com/user-attachments/assets/f05ff3c2-271f-4f7d-ad46-53b9a0419e4f" /> <br />

*subject to*

<img width="287" height="21" title = "x(k + 1) = x(k) + u(k); x(0) = b." alt="image" src="https://github.com/user-attachments/assets/23c66854-f885-4fe9-a31a-13c665f74cc1" /> <br />

## 5. A Linear Regression Problem Solved via Optimal Control

Let's consider a simple linear regression problem and let's first solve it using traditional linear least squares (LLS) and then using our optimal control based approach.
We have three (x, y) data points: *(1, 1), (2, 3), (3, 2).* We look for a line **y = m * x + b** that fits the data the best by minimizing the sum of squared residuals:

<img width="209" height="26" title = "S(m, b) = r _1 ^2 + r _2 ^2 + r _3 ^2 = " alt="image" src="https://github.com/user-attachments/assets/059e9d00-5280-4739-ada5-c56c1a77ec4e" />

<img width="334" height="24" title = "(m \cdot 1 + b - 1)^2 + (m \cdot 2 + b - 3)^2 + (m \cdot 3 + b - 2)^2 " alt="image" src="https://github.com/user-attachments/assets/06d5c16f-1578-47df-b1da-1c08ae62501c" />   <br />

**LLS gives us the following recipe for finding *m* and *b*:**

<img width="524" height="44" title = "\frac{\partial S}{\partial m} = 2 \cdot (m \cdot 1 + b - 1) \cdot 1 + 2 \cdot (m \cdot 2 + b - 3) \cdot 2 + 2 \cdot (m \cdot 3 + b - 2) \cdot 3 = 0;" alt="image" src="https://github.com/user-attachments/assets/820d2cf0-070f-48d5-a8eb-52232244bdd7" />

<img width="524" height="44" title = "\frac{\partial S}{\partial b} = 2 \cdot (m \cdot 1 + b - 1) \cdot 1 + 2 \cdot (m \cdot 2 + b - 3) \cdot 1 + 2 \cdot (m \cdot 3 + b - 2) \cdot 1 = 0." alt="image" src="https://github.com/user-attachments/assets/d4962e8b-be75-44b2-8ce0-e3a12e34a5d2" /> <br />

**Simple algebra yields the final answer:**

<img width="236" height="44" title = "m = \frac {1}{2}; b = 1; y = \frac {1}{2} \cdot x + 1" alt="image" src="https://github.com/user-attachments/assets/64b7227d-6f5d-4e30-a554-ec2d8ad1d4d0" /> <br />

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
