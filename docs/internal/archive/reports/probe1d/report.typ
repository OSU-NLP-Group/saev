// https://github.com/ryuryu-ymj/mannot
#import "@preview/mannot:0.3.0": *

#show link: underline
#set heading(numbering: "1.1.")

#align(center, text(17pt)[
  *Efficient Sparse 1D Probing*
])


= 1-D Case

For a latent $l$ and a class $c$, we fit a logistic regression classifier with parameters $theta = (b, w)$.
For a given example $j$, the raw prediction logit $eta_j$ is $b + w x_j$.
The probability $mu_j$ is $sigma(eta_j) = 1 / (1 + e^(-eta_j))$.
Our label $y_j$ is either 0 or 1.
Thus, our loss, with a L2 penalization (ridge) is:
$
cal(L)(b, w) = markhl(1/n sum_(j=1)^n [ -y_j log(mu_j) - (1 - y_j) log(1 - mu_j)], tag: #<eq:loss>) + lambda / 2 markhl((w^2 + (b - b_pi)^2 ), fill: #aqua, tag: #<eq:ridge>)

#annot(<eq:loss>, annot-text-props: (size: 1em))[Loss]
#annot(<eq:ridge>, annot-text-props: (fill: aqua, size: 1em))[Regularization]
$
#v(1em)

We regularize $b$ to $b_pi$, which is the baseline prevalence, and $w$ to 0 (predictions have no dependence on $x_j$).


#quote(
  block: true,
)[
== Intuition
First, approximate $cal(L)(theta)$ with a quadratic function using a Taylor series: $accent(cal(L), hat)(theta) = cal(L)(theta_k) + G_k^top (theta - theta_k) + 1/2 (theta - theta_k) H_k (theta - theta_k)$.
We know the first and second derivatives of binary cross entropy, so this is easy.

Second, choose $theta$ that minimize the quadratic approximation.
We take the derivative of our quadratic approximation $accent(cal(L), hat)$ with respect to $theta$ and find $theta$ that gives zero gradient (mimimum):
$
nabla_theta accent(cal(L), hat)(theta) &= 0 \
G_k + H_k (theta - theta_k) &= 0 \
(theta - theta_k) &= -H_k^(-1) G_k
$
The rest of this section describes the specific quantities and derivatives.
]



Let $s_j = mu_j (1 - mu_j)$.
Then our gradients are:
$
G
= mat(delim: "[", gap: #0.5em, g_0; g_1 )
= mat(delim: "[", gap: #0.5em, (diff cal(L)) / (diff b); (diff cal(L)) / (diff w))
= mat(delim: "[", gap: #0.5em, 1/n sum_j (mu_j - y_j) + lambda (b - b_pi); 1/n sum_j (mu_j - y_j)x_j + lambda w)
$
and our 2$times$2 Hessian is:
$
H = mat(
  delim: "[",
  gap: #0.5em,
  h_0, h_1;
  h_1, h_2;  
) = mat(
  delim: "[",
  gap: #0.5em,
  1/n sum_j s_j, 1/n sum_j s_j x_j;
  1/n sum_j s_j x_j, 1/n sum_j s_j x_j^2;
).
$
Once we calculate $G$ and $H$, we perform a *damped Newton step with a Levenberg–Marquardt-style diagonal shift*.
1. Form the gradient pieces $g_0$, $g_1$ and the clamped Hessian pieces $h_0$, $h_1$, $h_2$ and gradients.
2. Solve the 2$times$2 linear system $(H + lambda I) Delta = -G$ (where $Delta = theta - theta_k$) in closed form (see #link("https://en.wikipedia.org/wiki/Cramer%27s_rule")[Cramer's Rule]):#footnote([Recall that $Delta in RR^2$, and $Delta = mat(delim: "[", Delta b; Delta w)$.])
$
Delta b = (h_2 g_0 - h_1 g_1) / (det(H)), Delta w = (-h_1 g_0 + h_0 g_1) / (det(H)).
$
TODO: Add $lambda I$ to our closed form solution.
3. Compute the predicted reduction $g^top Delta - 1/2 Delta^top H Delta$. This is the classic TR/LM quadratic-model decrease (see #link("https://www.math.uci.edu/~qnie/Publications/NumericalOptimization.pdf")[UCI Mathematics]).

= Parallel


= Sparse CSR


Let:

- $N$ be the number of patches,
- $F$ be the number of SAE latents,
- $E$ be the number of non-zero values in a batch.

Let $bold(x) in RR^(N times F)$ be our training data. It is stored in a sparse CSR format.
Let $bold(w) in RR^(F times C)$ and $bold(b) in RR^(F times C)$ be our logistic regression parameters for every (latent, class) pair.

To optimize $bold(w)$ and $bold(b)$, we need to maintain $H_0, H_1, H_2, G_1, G_2 in RR^(F times C)$.
These matrices are aggregated over $N$.
We maintain some intermediate values as we iterate over $bold(x), bold(y)$ along $N$.
For a given class slab $C_b$, we iterate over all non-zero entries in $bold(x)$.
This row batch gives:
- $"latents" in {0,...,F-1}^E$
- $"values" in RR^E$
- $"row" in {0,...,N}^E$

We then calculate 

 and a given set of non-zero values $bold(x)_"nnz" in RR^(E times F)$
These are:
- $bold(mu) = bold(sigma) (bold(x)_"nnz" w  + b) in RR^("nnz" times F times C_b)$, the predict
