# Wasserstein distributionally robust logistic regression

Can a classifier that accounts for uncertainty in booking records predict hotel cancellations more reliably? This project implements standard logistic regression and a Wasserstein distributionally robust alternative in PyTorch, then evaluates both on the same held-out bookings.

## Results

The saved notebook reports 60,763 training bookings and 15,191 test bookings. At a classification threshold of 0.5, standard logistic regression improved on majority-class accuracy by 5.12 percentage points.

| Model | Accuracy | ROC-AUC | Cancellation precision | Cancellation recall |
| --- | ---: | ---: | ---: | ---: |
| Majority class | 73.08% | — | — | — |
| Logistic regression | 78.20% | 0.8254 | 0.6439 | 0.4257 |
| Wasserstein robust logistic regression | 77.41% | 0.8149 | 0.6556 | 0.3394 |

The robust model flagged cancellations with slightly higher precision but missed more of them. The stratified random split measures performance on similar bookings. Evaluating robustness to a change in hotels or booking periods would require a separate experiment.

## Implementation

- The pipeline cleans booking records, encodes categorical features and scales numeric inputs. Cross-validation fits the feature transformations on each training fold.
- The robust objective accounts for uncertainty in both features and labels. The optimizer enforces the L2 constraint on the model weights and dual variable during training.
- Five-fold validation compares ambiguity radii and label-change costs. The saved run selected a radius of 0.001 and a label-change cost of 2.

## Explore or reproduce

[Open the notebook](DR%20Logistic%20Regression/main.ipynb) to inspect the saved metrics, confusion matrices and evaluation plots. To train both models from the included CSV, run the following from the repository root in a Python virtual environment.

```bash
python -m pip install numpy "pandas>=2,<3" torch "scikit-learn>=1.2" matplotlib
cd "DR Logistic Regression"
python main.py
```
