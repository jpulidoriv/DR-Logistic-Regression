# Wasserstein distributionally robust logistic regression

This project compares two ways to predict whether a hotel booking will be canceled. Both use a linear classifier in PyTorch. Standard logistic regression minimizes the average loss on the training bookings. The robust model minimizes the worst-case expected loss over distributions within a Wasserstein distance $\varepsilon$ of the training distribution.

## Training objective

For booking $i$, the model computes a score $z_i = w^\top x_i + b$, where $y_i = 1$ denotes a cancellation. In the code’s 0/1 label convention, the logistic loss is

$$
\ell(z,y) = \log(1 + e^z) - yz.
$$

The implementation uses the dual formulation from [Shafieezadeh-Abadeh, Mohajerin Esfahani and Kuhn (2015)](https://arxiv.org/abs/1509.09259). For each booking, it compares the loss under the recorded label with the loss under the opposite label, adjusted for the cost $\kappa$ of changing a label:

$$
s_i = \max\bigl\{\ell(z_i,y_i),\ \ell(z_i,1-y_i)-\lambda\kappa\bigr\}.
$$

The resulting training objective is

$$
\min_{w,b,\lambda}\quad \lambda\varepsilon + \frac{1}{N}\sum_{i=1}^{N}s_i
\qquad \text{subject to} \qquad \lVert w\rVert_2 \leq \lambda.
$$

The training loop evaluates both losses with PyTorch’s binary cross-entropy on logits. After each gradient step, it projects the weights and the dual variable $\lambda$ back onto the constraint. The intercept $b$ is fitted separately from that projection.

## Hotel booking results

Five-fold cross-validation selected $\varepsilon = 0.001$ and $\kappa = 2$. Both models were then evaluated on the same 15,191 held-out bookings.

| Model | Accuracy | ROC-AUC | Cancellation precision | Cancellation recall |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 78.20% | 0.8254 | 64.39% | 42.57% |
| Wasserstein robust logistic regression | 77.41% | 0.8149 | 65.56% | 33.94% |

At the 0.5 decision threshold, the robust model was slightly more precise when it predicted a cancellation, but it identified fewer of the bookings that actually canceled. Standard logistic regression had the higher ROC-AUC on this split.

[Experiment notebook](DR%20Logistic%20Regression/main.ipynb)
