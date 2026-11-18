# Wasserstein distributionally robust logistic regression

This project compares two PyTorch models for predicting hotel booking cancellations. One uses standard logistic regression. The other minimizes the worst-case loss over nearby distributions, allowing for changes in both booking features and cancellation labels.

Five-fold cross-validation selects the robustness parameters. On 15,191 held-out bookings, standard logistic regression achieved a ROC-AUC of 0.825, compared with 0.815 for the robust model.
