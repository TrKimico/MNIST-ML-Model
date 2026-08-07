## Calculation Details
***Forward Propagation Equations***
\
Forward propagation uses the very common  computation equation:
$\hat{Z} =XW^{T}+B$
where $\hat{Z}$ is the output of the matrix multiplication of $X$(the input layer) and $W$(the weights connecting this layer to the next) which has to be transposed $W^{T}$ to fit the matrix multiplication constraints. Then a matrix addition is performed between this product and the $B$(the biases of the destination layer).
To obtain the activation value of each neuron, we need to pass this intermediate result through an activation function :
$a =ReLU(\hat{Z})$  for the two hidden layers, and $a =SoftMax(\hat{Z})$ for the output layer.

***Back Propagation Equations***
\
To compute back propagation, we must first apply the Gradient Descent partial derivative formula and apply it to our specific case (ReLU and Soft-Max activation) in order to get the initial Cost signal of the output layer, which will be used to compute the cost of the parameters attached to it, and propagated via the transposed weight matrix.
So, the standard function :

$\dfrac{\partial C}{\partial w^{(L)}} =\dfrac{\partial z^{(L)}}{\partial w^{(L)}} \cdot \dfrac{\partial a^{(L)}}{\partial z^{(L)}} \cdot \sum_{k=1}^{n^{L+1}}\dfrac{\partial C}{\partial a_{k}^{(L+1)}}$

In practice becomes:

$$
\nabla_w C = \begin{bmatrix}
\text{vec}\left(\dfrac{\partial C}{\partial w^{(n_0)}}\right) \\
\text{vec}\left(\dfrac{\partial C}{\partial w^{(n_1)}}\right) \\
\text{vec}\left(\dfrac{\partial C}{\partial w^{(n_2)}}\right)
\end{bmatrix}
$$

with:

$$\dfrac{\partial C}{\partial w^{(n_0)}} = a_i^{(\text{input})} \cdot \text{ReLU}'(z_j^{(n_0)}) \cdot \sum_k \left[ w_{jk}^{(n_1)} \cdot \text{error}_k^{(n_1)} \right]$$

$$\dfrac{\partial C}{\partial w^{(n_1)}} = a_i^{(n_0)} \cdot \text{ReLU}'(z_j^{(n_1)}) \cdot \sum_k \left[ w_{jk}^{(n_2)} \cdot \text{error}_k^{(n_2)} \right]$$

$$\dfrac{\partial C}{\partial w^{(n_2)}} = a_i^{(n_1)} \cdot (\hat{y}_j - y_j)$$

and

$$
\nabla_b C = \begin{bmatrix}
\dfrac{\partial C}{\partial b^{(n_0)}} \\
\dfrac{\partial C}{\partial b^{(n_1)}} \\
\dfrac{\partial C}{\partial b^{(n_2)}}
\end{bmatrix}
$$

with:

$$\dfrac{\partial C}{\partial b^{(n_0)}} = \text{ReLU}'(z_j^{(n_0)}) \cdot \sum_k \left[ w_{jk}^{(n_1)} \cdot \text{error}_k^{(n_1)} \right]$$

$$\dfrac{\partial C}{\partial b^{(n_1)}} = \text{ReLU}'(z_j^{(n_1)}) \cdot \sum_k \left[ w_{jk}^{(n_2)} \cdot \text{error}_k^{(n_2)} \right]$$

$$\dfrac{\partial C}{\partial b^{(n_2)}} = \delta^{(n_2)}_j = \hat{y}_j - y_j$$

finally:

$$
\nabla_\delta C = \begin{bmatrix}
\delta^{(n_0)} \\
\delta^{(n_1)} \\
\delta^{(n_2)}
\end{bmatrix}
$$

with:

$$\delta^{(n_0)}_j = \sum_k \left[ w_{jk}^{(n_1)} \cdot \text{error}_k^{(n_1)} \right]$$

$$\delta^{(n_1)}_j = \sum_k \left[ w_{jk}^{(n_2)} \cdot \text{error}_k^{(n_2)} \right]$$

$$\delta^{(n_2)}_j = \hat{y}_j - y_j$$
