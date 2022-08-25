# Question 2
## Part A

$$ \bigtriangledown_xf(x) \= \bigtriangledown_x \left( b^Tx + x^TAx\right)$$

Lets take the gradient of the first term

$$ \bigtriangledown_x b^Tx \= b^T \=  \begin{pmatrix}b1\\ b2 \\ ...\\ bn\end{pmatrix}^T $$

Lets take the gradient of the second term

$$ \bigtriangledown_x \left(x^TAx\right) \= \begin{bmatrix} \sum _{i=1}^n a_{i1}x_i + \sum\_{i=1}^n \left(a_{1i}x_i \right) && ...&& \sum_{i=1}^n \left(a_{ni}x_i\right) + \sum_{i=1}^n \left(a_{in}x_i \right) \end{bmatrix}^T $$
 

## Part B
## Part C
ALL eigen values must be positive
## Part D
All rows are linearly independent, meaning all eigenvalues are not equal to 0. Therefore:

$$ \|A\| \neq 0 $$

## Part E
b must be perpendicular to y as seen here:

  Given:
 
 $$ A^T y \= 0 $$
 
 $$ Ax \= b$$ 
 
 we know 
 
 $$\left(A^Ty \right)^T \= 0^T \rightarrow y^TA \= 0^T $$
 
 $$ y^TAx \= y^Tb $$
 
 $$ 0^T x \= y^T b $$
 
   In order for a column vector multiplied by a row vector to be equal to zero, the 2 vextors must be parrallel to one another.
# Question 3

minimize:

$$\sum_{i=1}^N c_i x_i  $$

S.T.:

$$ \sum_{i=1}^N a_{ij} x_i \geq b_j  $$ 

for all j = 1, 2, ... , M

