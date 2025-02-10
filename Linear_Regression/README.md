In this notebook, we will implement a simple linear regression model using NumPy. Linear regression is a fundamental technique in machine learning and statistics used to model the relationship between a dependent variable and an independent variable.  

We will follow these steps to build our model:  
1. **Create a dataset** – Generate synthetic data for our regression task.  
2. **Describe the model** – Define the equation of a simple linear regression.  
3. **Describe the cost function** – Introduce the mean squared error (MSE) to evaluate model performance.  
4. **Gradient descent** – Compute the gradients needed for optimization and optimize the model parameters using gradient descent.  

By the end of this notebook, we will have a working linear regression model implemented from scratch using NumPy.

### 1. Dataset Representation  

We create a dataset $(x, y)$ with $m = 100$ rows, $n = 1$ feature $x$, and $y$ as the target variable.

$$
\begin{array}{|c|c|}
\hline
x & y \\
\hline
x^{(1)} & y^{(1)} \\
x^{(2)} & y^{(2)} \\
\vdots & \vdots \\
x^{(m)} & y^{(m)} \\
\hline
\end{array}
$$

### 2. Describe the Model  

Our model is represented by the function:  
$$ f(x) = ax + b $$

In **matrix form**, we express this as:  
$$ F = X \theta $$

where:  

- $F$ is the predicted output, represented as a column vector:  
  $$
  F = \begin{bmatrix} f(x^{(1)}) \\ f(x^{(2)}) \\ \vdots \\ f(x^{(m)}) \end{bmatrix}
  $$
  with $\dim(F) = (m \times 1)$.  

- $X$ is the feature matrix, including a column of ones for the bias term:  
  $$
  X = \begin{bmatrix} x^{(1)} & 1 \\ x^{(2)} & 1 \\ \vdots & \vdots \\ x^{(m)} & 1 \end{bmatrix}
  $$
  with $\dim(X) = m \times (n+1) = m \times 2$

- $\theta$ is the parameter vector:  
  $$
  \theta = \begin{bmatrix} a \\ b \end{bmatrix}
  $$
  with $\dim(\theta) = (n+1) \times 1 = 2 \times 1$.  

### 3. Describe the Cost Function  

The cost function $$J$$ is given by the mean squared error ([MSE](https://en.wikipedia.org/wiki/Mean_squared_error)):

$$
J(a, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( ax^{(i)} + b - y^{(i)} \right)^2 \tag{1}
$$

In **matrix form**:

$$
J(\theta) = \frac{1}{2m} (X\theta - Y)^T (X\theta - Y) \tag{2}
$$

where $$Y$$ is a column vector:

$$
Y = \begin{bmatrix} y^{(1)} \\ y^{(2)} \\ \vdots  \\ y^{(m)}\end{bmatrix}
$$

### 4. Gradient Descent  

Gradient Descent is an optimization algorithm used to minimize the cost function $$J(a, b)$$ by iteratively updating the parameters $$a$$ and $$b$$ as represented here for parameter $$a$$:

![Gradient Descent](gradient_descent.jpg)

The update rules for the parameters are given by:  

$$
a_{i+1} = a_{i} - \alpha \frac{\partial J}{\partial a} \tag{3}
$$

$$
b_{i+1} = b_{i} - \alpha \frac{\partial J}{\partial b} \tag{4}
$$


where:  
- $\alpha$ is the learning rate, controlling the step size and is greater than 0.  
- $\frac{\partial J}{\partial a}$ and $\frac{\partial J}{\partial b}$ are the gradients of the cost function with respect to $a$ and $b$.  

The process is repeated until convergence, ensuring the model parameters minimize the cost function.

The gradients are computed as follows:

$$
\frac{\partial J(a,b)}{\partial a} = \frac{1}{m} \sum_{i=1}^{m} \left( ax^{(i)} + b - y^{(i)} \right) x^{(i)} \tag{5}
$$

$$
\frac{\partial J(a,b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( ax^{(i)} + b - y^{(i)} \right) \tag{6}
$$

In **matrix form**, we define $\frac{\partial J(\theta)}{\partial \theta}$ as:

$$
\frac{\partial J(\theta)}{\partial \theta} =
\begin{bmatrix}
    \frac{\partial J(a,b)}{\partial a} \\ \\
    \frac{\partial J(a,b)}{\partial b}
\end{bmatrix}
= \frac{1}{m}X^T(X\theta - Y) \tag{7}
$$

which has dimension $(n+1) \times 1$.

