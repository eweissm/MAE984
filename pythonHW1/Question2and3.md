# Question 2
## Part A

$$ \bigtriangledown_xf(x) \= \bigtriangledown_x \left( b^Tx + x^TAx\right)$$

Lets take the gradient of the first term

$$ \bigtriangledown_x b^Tx \= b^T \=  \begin{pmatrix}
b1\\
b2\\ 
...\\
bn \end{pmatrix}$$

Lets take the gradient of the second term

$$ \bigtriangledown_x \left(x^TAx\right) \=
\begin{pmatrix} \sum_{i=1}^n \left(a_{1i\:}x_i\right)+\sum \:_{i=1}^n\left(a_{i1\:}x_i\right)\\
...\\
\sum \:_{i=1}^n\left(a_{ni}\:x_i\right)+\sum \:\:_{i=1}^n\left(a_{n1\:}x_i\right)\end{pmatrix} $$
 
 therefore
 
 $$ \bigtriangledown_xf(x) \= \bigtriangledown_x \left( b^Tx + x^TAx\right) \= \begin{pmatrix}
b1\\
b2\\ 
...\\
bn \end{pmatrix}
+\begin{pmatrix}\sum _{i=1}^n\left(a_{1i\:}x_i\right)+\sum \:_{i=1}^n\left(a_{i1\:}x_i\right)\\
...\\
\sum \:_{i=1}^n\left(a_{ni}\:x_i\right)+\sum \:\:_{i=1}^n\left(a_{n1\:}x_i\right)\end{pmatrix} $$
$$


## Part B
Similar to Part A, we will first take the hessian of the first term:

$$ H(b^T) = 0 $$

because the derivative of a constant = 0

then for the second part 

$$ H(\begin{pmatrix}\sum _{i=1}^n\left(a_{1i\:}x_i\right)+\sum \:_{i=1}^n\left(a_{i1\:}x_i\right)\\
...\\
\sum \:_{i=1}^n\left(a_{ni}\:x_i\right)+\sum \:\:_{i=1}^n\left(a_{n1\:}x_i\right)\end{pmatrix})
   \= \begin{pmatrix}a_{11}&a_{21}+a_{12}&...&a_{1n}+a_{n1}\\
a_{21}+a_{12}&a_{22}&...&a_{2n}+a_{n2}\\
...&...&...&\\
a_{1n}+a_{n1}&a_{2n}+a_{n2}&&a_{nn}\end{pmatrix}$$

therefore,

$$H(f(x)) \= \begin{pmatrix}a_{11}&a_{21}+a_{12}&...&a_{1n}+a_{n1}\\
a_{21}+a_{12}&a_{22}&...&a_{2n}+a_{n2}\\
...&...&...&\\
a_{1n}+a_{n1}&a_{2n}+a_{n2}&&a_{nn}\end{pmatrix}$$

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

and

$$ x \geq 0 $$

