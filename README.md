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

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/5ef15d28-e71c-4cdb-a30c-18e0dabe4f84"> 
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/1a074ad8-e46a-47c4-923b-66824b19e362">  
  <img width="56" height="39" title = "\bbox[black]{\color{white}\frac{dx}{dt} = u}" alt="fallback image" src="https://github.com/user-attachments/assets/5ef15d28-e71c-4cdb-a30c-18e0dabe4f84">
</picture>      <br />   <br />  

with the initial condition: **x(0) = b**.

We can also discretize it (e.g. with 1 second time step) to obtain (think an Euler like forward approximation):

**x(k+1) = x(k) + u(k)**, where **u(k)= u(k+1) = const.**

Since *u* is constant, it can also be written as:

**x(k) = x(0) + u * k.**


An optimization problem can be thought of as finding an optimal control *u* (the line's slope) such that it minimizes the sum of squared errors between predicted and measured states. In other words we want to minimize the following cost functional (a *Bolza* problem, with the *fixed time/free final point* condition):

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/262855c3-612b-4c66-891d-4e26901b04bd"> 
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/0a55c3cf-6d9a-403f-bdc4-3bf46e738bcd">  
  <img width="268" height="19" title = "\bbox[black]{\color{white}J = running\,cost + terminal\,cost =}" alt="fallback image" src="https://github.com/user-attachments/assets/262855c3-612b-4c66-891d-4e26901b04bd">
</picture>      <br /> 

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/9b421efb-ba39-4383-907e-fcdb66ab0979"> 
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/66e930bc-52f8-4611-972f-0024c034c4f0">  
  <img width="308" height="51" title = "\bbox[black]{\color{white}\frac{1}{2}\int_{0}^{T}(x(t) - z(t))^2 \mathrm{d}t +\frac{1}{2}g\cdot(x(T) - z(T))^2}" alt="fallback image" src="https://github.com/user-attachments/assets/9b421efb-ba39-4383-907e-fcdb66ab0979">
</picture>      <br />   <br />

where *z(t)* can be approximated, say, by a spline function (spline interpolation) passing through a set of N data points (coordinate pairs of *(t, z(t)).* And *x(t)* is satisfying the dynamic constraint:

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/92c1d1a4-2ab2-43a1-87ad-e6474af19ed5"> 
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/f4ee00ae-2b21-4f1e-b5c2-9d060374f0f0">  
  <img width="56" height="18" title = "\bbox[black]{\color{white}\dot{x} = u}" alt="fallback image" src="https://github.com/user-attachments/assets/92c1d1a4-2ab2-43a1-87ad-e6474af19ed5">
</picture>   <br /> <br />

Note, nowhere in our calculations we will ever need the explicit formula for *z(t).* We only interested in z's values at the N data points:
*z(1), z(2), z(3), ..., z(N).* We also use what's called a *"soft"* terminal constraint with an adjustable weight *g.* 

**Unlike the [Linear Quadratic] Tracking Control problem (or similar trajectory optimization tasks) we do not follow *z(t)* (aka "reference trajectory"). Instead our system always stays on a straight line, and we only control the slope.**

We form a Hamiltonian (also known as Pontryagin's Hamiltonian) which is typically the sum of the running cost and the costate times dynamics:

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/2ec78d19-5ae5-4b7d-866e-8163d6866538"> 
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/6a3933bb-f1a8-49c3-8dd5-b0a1603b95ba">  
  <img width="304" height="42" title = "\bbox[black]{\color{white}\ H(x, u, \lambda, t) = \frac{1}{2}(x(t) - z(t))^2 + \lambda(t) \cdot u}" alt="fallback image" src="https://github.com/user-attachments/assets/2ec78d19-5ae5-4b7d-866e-8163d6866538" >
</picture>   <br /> <br />

Since (in continuous case):

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/910b8560-f47e-4154-aca3-201aeb67c74e"> 
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/47626c44-de6b-4037-9ac1-ed639673d6ad">  
  <img width="158" height="22" title = "x(t) = x(0) + u\cdot t" alt="fallback image" src="https://github.com/user-attachments/assets/910b8560-f47e-4154-aca3-201aeb67c74e" >
</picture>   <br /> <br />

the expression for Pontryagin's Hamiltonian becomes:

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/31ec249a-ae79-4923-9c51-ab15a9d0fb6e"> 
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/8979626e-1838-49b3-a9d7-428af7c7bf4c">  
  <img width="351" height="42" title ="\bbox[black]{\color{white}\ H(u, \lambda, t) = \frac{1}{2}(x(0) + u \cdot t - z(t))^2 + \lambda(t) \cdot u}" alt="fallback image" src="https://github.com/user-attachments/assets/31ec249a-ae79-4923-9c51-ab15a9d0fb6e" >
</picture>   <br /> <br />

Then for the costate equation we obtain:

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/a837f9aa-9ed4-47f2-bd99-96072950aaeb"> 
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/57085eb3-de6a-41c4-ac5e-1c04ca47f436">  
  <img width="124" height="42" title = "\bbox[black]{\color{white}\frac{d \lambda}{dt} = - \frac{\partial H}{\partial x} = 0}" alt="fallback image" src="https://github.com/user-attachments/assets/a837f9aa-9ed4-47f2-bd99-96072950aaeb">
</picture>   <br /> <br />

Therefore, the costate is constant and we can find its value using the terminal cost condition:

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/8bb4a903-43fd-44c7-ace2-432fc9762ecb"> 
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/cb9c5fa3-bfbe-4ebe-8c1b-5971f81479ab">  
  <img width="400" height="44" title = "\bbox[black]{\color{white}\lambda(t) = \lambda(T)  = \frac{\partial (\frac{1}{2}g\cdot(x(T) - z(T))^2)}{\partial x} = g \cdot (x(T) - z(T))}" alt="fallback image" src="https://github.com/user-attachments/assets/8bb4a903-43fd-44c7-ace2-432fc9762ecb" >
</picture>   <br /> <br />

Minimization of the Hamiltonian with respect to *u* yields:

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/0c0e62c1-33b6-410d-a263-76137e9048d5"> 
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/86f7d352-c0a1-41ca-8464-18281278a336">  
  <img width="580" height="40" title = "\bbox[black]{\color{white}\frac{\partial H}{\partial u} = (x(0) + u \cdot t - z(t)) \cdot t + \lambda (t) = (x(0) + u \cdot t - z(t)) \cdot t + g \cdot (x(T) - z(T)) = 0}" alt="fallback image" src="https://github.com/user-attachments/assets/0c0e62c1-33b6-410d-a263-76137e9048d5" >
</picture>   <br /> <br />

After discretizing the cost functional and the dynamics we obtain the following optimization problem:

*minimize*

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/ddb6f054-355e-4668-9e7d-337a7f502900"> 
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/7db73ea0-9fe9-49e2-8212-e3199119ebe6">  
  <img width="325" height="61" title = "\bbox[black]{\color{white}J = \frac{1}{2}\sum_{k=1}^{T-1}(x(k) - z(k))^2 + \frac{1}{2} g\cdot(x(T) - z(T))^2}" alt="fallback image" src="https://github.com/user-attachments/assets/ddb6f054-355e-4668-9e7d-337a7f502900" >
</picture>   <br /> <br />

*subject to*

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/8e0b3eef-8b5f-4f8c-b529-2896e39dc36b"> 
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/aa348cb3-285d-45ef-ab68-0ef29e96df52">  
  <img width="288" height="22" title = "\bbox[black]{\color{white}x(k + 1) = x(k) + u(k); x(0) = b.}" alt="fallback image" src="https://github.com/user-attachments/assets/8e0b3eef-8b5f-4f8c-b529-2896e39dc36b">
</picture>   <br /> <br />



## 5. A Linear Regression Problem Solved via Optimal Control

Let's consider a simple linear regression problem and let's first solve it using traditional linear least squares (LLS) and then using our optimal control based approach.
We have three (x, y) data points: *(1, 1), (2, 3), (3, 2).* We look for a line **y = m * x + b** that fits the data the best by minimizing the sum of squared residuals:

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/78c36dfe-bb37-465d-b6b6-4bc3352b3b5a"> 
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/dae18612-d793-44c8-8305-e3e0e9b45d72">  
  <img width="211" height="28" title = "\bbox[black]{\color{white}S(m, b) = r _1 ^2 + r _2 ^2 + r _3 ^2 = }" alt="fallback image" src="https://github.com/user-attachments/assets/78c36dfe-bb37-465d-b6b6-4bc3352b3b5a" >
</picture>    <br /> 

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/30432e65-2841-4963-884a-d83d86b233af"> 
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/fcae7956-330d-4eb4-a48d-f60bafad8ec2">  
  <img width="335" height="25" title = "\bbox[black]{\color{white}(m \cdot 1 + b - 1)^2 + (m \cdot 2 + b - 3)^2 + (m \cdot 3 + b - 2)^2 }" alt="fallback image" src="https://github.com/user-attachments/assets/30432e65-2841-4963-884a-d83d86b233af" >
</picture>    <br /> <br />


**LLS gives us the following recipe for finding *m* and *b*:**

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/efe6475a-67c9-4757-a91c-16924dde9336"> 
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/8c75db6f-158e-4426-86d2-e54e1f01f321">  
  <img width="524" height="43" title = "\bbox[black]{\color{white}\frac{\partial S}{\partial m} = 2 \cdot (m \cdot 1 + b - 1) \cdot 1 + 2 \cdot (m \cdot 2 + b - 3) \cdot 2 + 2 \cdot (m \cdot 3 + b - 2) \cdot 3 = 0;}" alt="fallback image" src="https://github.com/user-attachments/assets/efe6475a-67c9-4757-a91c-16924dde9336" >
</picture>   <br />    <br />


<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/db5f7030-17e0-4822-9948-54e9a2f4e65e"> 
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/86bf73c7-ca3b-4752-977b-cfd6062a9cdc">  
  <img width="524" height="43" title = "\bbox[black]{\color{white}\frac{\partial S}{\partial b} = 2 \cdot (m \cdot 1 + b - 1) \cdot 1 + 2 \cdot (m \cdot 2 + b - 3) \cdot 1 + 2 \cdot (m \cdot 3 + b - 2) \cdot 1 = 0.}" alt="fallback image" src="https://github.com/user-attachments/assets/db5f7030-17e0-4822-9948-54e9a2f4e65e" >
</picture>    <br />    <br />


**Simple algebra yields the final answer:**

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/78797cc5-0e0d-412e-8a43-cdebadd02307"> 
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/143b9528-6066-4ec9-862c-29325873d16e">  
  <img width="236" height="43" title = "\bbox[black]{\color{white}m = \frac {1}{2}; b = 1; y = \frac {1}{2} \cdot x + 1}" alt="fallback image" src="https://github.com/user-attachments/assets/78797cc5-0e0d-412e-8a43-cdebadd02307"> 
</picture>    <br />    <br />


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

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/338739a4-d47e-41d1-a722-2d69a9499d71"> 
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/321b20ab-3442-409c-937e-293a6b517baa">  
  <img width="29" height="41" title = "\bbox[black]{\color{white}\frac{\partial H}{\partial u}}" alt="fallback image" src="https://github.com/user-attachments/assets/338739a4-d47e-41d1-a722-2d69a9499d71" > 
</picture>    <br />    <br />

and expanding the gradient of the cost function:

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/875de9d4-7aed-4d70-854c-8006ecc4e982"> 
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/291013bb-3aa4-4e8c-b7cd-1abb8047380a">  
  <img width="600" height="58" title = "\bbox[black]{\color{white}\nabla _u J = \sum_{k=1}^{T-1} \frac{\partial H}{\partial u(k)} + g \cdot (x(T) - z(T)) = \sum_{k=1}^{T-1} (x(0) + u \cdot k - z(k)) \cdot k) + g \cdot (x(T) - z(T)) = 0}" alt="fallback image" src="https://github.com/user-attachments/assets/875de9d4-7aed-4d70-854c-8006ecc4e982" >
</picture>    <br />    <br />

we obtain:

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/d68c0048-f2e1-4a31-9a78-54e66601f149"> 
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/b51e5ba4-7fde-44cd-807d-48102707923c">  
  <img width="485" height="22" title = "\bbox[black]{\color{white}(x(0) + u \cdot 1 - 1) \cdot 1+ (x(0) + u \cdot 2 - 3) \cdot 2 + g \cdot (x(0) + u \cdot 3 - 2) = 0}" alt="fallback image" src="https://github.com/user-attachments/assets/d68c0048-f2e1-4a31-9a78-54e66601f149" > 
</picture>    <br />    <br />

and if we let *x(0) = b = 1* and *g = 3* it again yields the same final answer for the slope *u*:

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/efa80183-caad-476a-aa99-2eb67e8e95ec"> 
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/a4070b2b-9905-47a1-8fdf-99a91ac80f71">  
  <img width="197" height="44" title = "\bbox[black]{\color{white}u = \frac {1}{2}; x(t) = x(0) + \frac {1}{2} \cdot t}" alt="fallback image" src="https://github.com/user-attachments/assets/efa80183-caad-476a-aa99-2eb67e8e95ec" >
</picture>    <br />    <br />



## References

<a id="1">[1]</a>
Stephen Boyd, (2009) [Lecture 1. Linear quadratic regulator: Discrete-time finite horizon](https://stanford.edu/class/ee363/lectures/dlqr.pdf)

<a id="2">[2]</a>
Arthur E. Bryson, Jr., Yu-Chi Ho, (1975) Applied Optimal Control: Optimization, Estimation and Control, Chapter 2

<a id="3">[3]</a>
Emanuel Todorov, (2006) [Optimal Control Theory](https://roboti.us/lab/papers/TodorovChapter06.pdf)

<a id="4">[4]</a>
Ben Recht, (2016) [Mates of Costate](https://archives.argmin.net/2016/05/18/mates-of-costate/)
