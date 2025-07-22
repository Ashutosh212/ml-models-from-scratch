# ML Models From Scratch

This repository contains implementations of fundamental machine learning algorithms from scratch using Python and NumPy. It's designed for learning purposes and interview preparation.

## Implemented Algorithms

- ☑️ K-Means Clustering
- ☑️ Linear Regression
- Logistic Regression
- Decision Trees 
- Support Vector Machine
- SVD
- PCA
- Neural Networks 

### Linear Regression Evaluation

#### Dataset: Single Feature (Attendance Hours → Final Marks)
| Model                    | MSE      | R² Score |
|--------------------------|----------|----------|
| Custom Linear Regression | 0.1725   | 0.8234   |
| Sklearn SGDRegressor     | 0.1725   | 0.8233   |

#### Dataset: Multiple Features — Graduate Admission Prediction
| Model                    | MSE      | R² Score |
|--------------------------|----------|----------|
| Custom Linear Regression | 0.5535   | 0.5652   |
| Sklearn SGDRegressor     | 0.2349   | 0.8155   |

### Logistic Regression Evaluation (User Metadata -> Purchase Prediction)

| Model                      | MSE                 | R² Score |
|----------------------------|---------------------|----------|
| Custom Logistic Regression | 3.13 × 10⁻³⁰        | 1.0      |
| Sklearn Logistic Regression| 0.0                 | 1.0      |

