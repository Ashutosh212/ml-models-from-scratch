# Logistic Regression: Likelihood, Loss Function, and Sigmoid

---

## Sigmoid Function (Logistic Function)

The sigmoid function maps any real-valued number to a value between 0 and 1:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

In logistic regression:

\[
\hat{y} = \sigma(w^\top x + b)
\]

---

## Bernoulli Likelihood for Logistic Regression

Assuming a binary classification task, for a single training example \((x_i, y_i)\), where \(y_i \in \{0, 1\}\), the likelihood is:

\[
P(y_i \mid x_i; \theta) =
\begin{cases}
\hat{y}_i & \text{if } y_i = 1 \\
1 - \hat{y}_i & \text{if } y_i = 0
\end{cases}
\]

This can be written compactly as:

\[
P(y_i \mid x_i; \theta) = \hat{y}_i^{y_i} (1 - \hat{y}_i)^{1 - y_i}
\]

---

## Log-Likelihood Function

Assuming \(N\) independent examples:

\[
\mathcal{L}(\theta) = \prod_{i=1}^{N} \hat{y}_i^{y_i} (1 - \hat{y}_i)^{1 - y_i}
\]

Taking the log to simplify (log-likelihood):

\[
\log \mathcal{L}(\theta) = \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
\]

---

## Logistic Regression Loss Function (Cross-Entropy)

We define the **loss function** as the **negative log-likelihood**, which we minimize:

\[
\mathcal{J}(\theta) = - \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
\]

Or for average loss:

\[
\mathcal{J}(\theta) = - \frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
\]

---

## Optimization Objective

- Maximize the **log-likelihood**  
  or  
- Minimize the **cross-entropy loss**

Gradient descent is used to find optimal parameters \( \theta = (w, b) \).

---

## Gradient (Optional for Optimization)

The gradient w.r.t. weights for one sample is:

\[
\frac{\partial \mathcal{J}}{\partial w} = ( \hat{y}_i - y_i ) x_i
\]

This guides the update rule during training.

---
### Derivative of Logistic Regression Loss w.r.t. Weights

Given the loss:
\[
\mathcal{J}_i(w) = -\left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
\]
And prediction:
\[
\hat{y}_i = \sigma(z_i) = \frac{1}{1 + e^{-w^\top x_i}}
\]

Using the chain rule:
\[
\frac{\partial \mathcal{J}_i}{\partial w} = \frac{d\mathcal{J}_i}{d\hat{y}_i} \cdot \frac{d\hat{y}_i}{dz_i} \cdot \frac{dz_i}{dw}
\]

Leads to:
\[
\frac{\partial \mathcal{J}_i}{\partial w} = (\hat{y}_i - y_i) x_i
\]

---

**Summary**:
- Use **sigmoid** to squash linear output into \([0, 1]\)
- Assume **Bernoulli distribution** over binary labels
- Use **log-likelihood** for model fit
- **Minimize cross-entropy** as the loss function

