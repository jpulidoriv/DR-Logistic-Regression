# This project aims to predict whether a hotel booking will be cancelled
# based on information available at the time of booking.
#
# It implements a Distributionally Robust Optimization (DRO) approach
# to Logistic Regression to improve robustness against shifts in the
# underlying data distribution.


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

# Data cleaning

df = pd.read_csv("hotel_bookings.csv")

# Examine data before cleaning
missing_values = {}
for column in df.columns:
    mv = df[column].isna().sum()
    if mv > 0:
        missing_values[column] = int(mv)
print(missing_values)


df["company"] = df["company"].fillna("no_company")
df["agent"] = df["agent"].fillna("no_agent")
df["country"] = df["country"].fillna("unknown")

#Filter the rows where an amount is negative
clean_df = df.copy()
rows = df[(df["adr"] < 0)| (df["children"].isna())]
clean_df = clean_df.drop(rows.index)


#Convert hotel into binary
clean_df["hotel"] = clean_df["hotel"].replace({
    "Resort Hotel": 0,
    "City Hotel": 1
    })


#Drop columns
clean_df = clean_df.drop(columns=["arrival_date_week_number",
                                  "assigned_room_type",
                                  "booking_changes", 
                                  "days_in_waiting_list", 
                                  "deposit_type",
                                  "reservation_status",
                                  "reservation_status_date"])


#Change information about week of the month of arrival
clean_df["arrival_week_of_month"] = pd.cut(
    clean_df["arrival_date_day_of_month"],
    bins=[0, 7, 14, 21, 31],
    labels=["first_week", "second_week", "third_week", "fourth_week"]
)
clean_df = clean_df.drop(columns=["arrival_date_day_of_month"])
clean_df["arrival_date_month"] = clean_df["arrival_date_month"].str.strip().str.lower()


#Remove rows with 0 stays
zero_nights = clean_df[clean_df["stays_in_weekend_nights"] + clean_df["stays_in_week_nights"] <= 0]
clean_df = clean_df.drop(zero_nights.index)


#Remove rows with 0 guests
zero_guests = clean_df[(clean_df["adults"] + clean_df["children"] + clean_df["babies"]<= 0) | (clean_df["adults"] == 0)]
clean_df = clean_df.drop(zero_guests.index)
clean_df["children"] = clean_df["children"].astype(int)
print(clean_df["children"].dtype)


clean_df["meal"] = clean_df["meal"].replace({
    "Undefined": "SC"
}).str.strip().str.lower()
print(clean_df["meal"].value_counts())


clean_df["country"] = clean_df["country"].str.strip().str.lower()
clean_df["market_segment"] = clean_df["market_segment"].str.strip().str.lower()
clean_df["distribution_channel"] = clean_df["distribution_channel"].str.strip().str.lower()
clean_df["reserved_room_type"] = clean_df["reserved_room_type"].str.strip().str.lower()
clean_df["agent"] = clean_df["agent"].astype(str)
clean_df["company"] = clean_df["company"].astype(str)
clean_df["customer_type"] = clean_df["customer_type"].str.strip().str.lower()


df1 = clean_df[clean_df["adr"] == 5400]
clean_df = clean_df.drop(df1.index)

clean_df = clean_df[
    (clean_df["adults"] <= 2) &
    ((clean_df["adults"] + clean_df["children"] + clean_df["babies"]) <= 4)
].copy()

clean_df = clean_df[
    (clean_df["distribution_channel"] != "undefined") &
    (clean_df["required_car_parking_spaces"] <= 2) &
    ((clean_df["stays_in_weekend_nights"] + clean_df["stays_in_week_nights"]) <= 30)
].drop_duplicates().reset_index(drop=True)


inconsistent_rows = clean_df[
    ((clean_df["is_repeated_guest"] == 0) & ((clean_df["previous_cancellations"] > 0) |(clean_df["previous_bookings_not_canceled"] > 0)))
    |
    (
        (clean_df["is_repeated_guest"] == 1) &
        (clean_df["previous_cancellations"] == 0) &
        (clean_df["previous_bookings_not_canceled"] == 0)
    )
]

clean_df = clean_df.drop(inconsistent_rows.index)

clean_df["country"] = clean_df["country"].replace({
    "cn": "chn",
    "tmp": "tls"
})

clean_df = clean_df.drop(columns=[
    "required_car_parking_spaces"
])

clean_df = clean_df.drop_duplicates().reset_index(drop=True)


#Remove rows with unknown country
rows_unk_country = clean_df[clean_df["country"] == "unknown"]
clean_df = clean_df.drop(rows_unk_country.index)


#Remove room type l
clean_df = clean_df[
    clean_df["reserved_room_type"] != "l"
].copy()

# Categorical columns for OHE
categorical_columns = [
    "arrival_date_year",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "agent",
    "company",
    "customer_type",
    "arrival_week_of_month"
]

# Train/Test Split

X = clean_df.drop(columns=["is_canceled"])
y = clean_df["is_canceled"]
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=y)

# Perform 5 fold cross validation

normal_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

#Encoder for columns where their categories require frequency >500
frequent_encoder = OneHotEncoder(
    min_frequency=501,
    handle_unknown="infrequent_if_exist",
    sparse_output=False
)

company_encoder = OneHotEncoder(
    min_frequency=81,
    handle_unknown="infrequent_if_exist",
    sparse_output=False
)

normal_columns = [
    "arrival_date_year",
    "arrival_date_month",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "customer_type",
    "arrival_week_of_month"
]

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

numerical_columns = [
    col for col in X_train.columns
    if col not in categorical_columns
    and col not in ["hotel", "is_repeated_guest"]
]

preprocessor = ColumnTransformer(
    transformers=[
        ("normal", normal_encoder, normal_columns),
        ("frequent", frequent_encoder, ["country", "agent"]),
        ("company", company_encoder, ["company"]),
        ("numerical", StandardScaler(), numerical_columns)
    ],
    remainder="passthrough"  #Important for keeping the columns we don't transofrm
)

val_losses = []
val_accuracies = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
    # 1. Select training and validation data
    X_fold_train = X_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]

    y_fold_train = y_train.iloc[train_idx]
    y_fold_val = y_train.iloc[val_idx]

    # 2. Preprocessing
    X_fold_train_transformed = preprocessor.fit_transform(X_fold_train)
    X_fold_val_transformed = preprocessor.transform(X_fold_val)

    # 3. Convert training data to tensors
    X_tensor = torch.tensor(
        np.asarray(X_fold_train_transformed, dtype=np.float32)
    )

    y_tensor = torch.tensor(
        y_fold_train.to_numpy(),
        dtype=torch.float32
    ).reshape(-1, 1)

    # 4. Create a NEW model for each fold
    model = nn.Linear(X_tensor.shape[1], 1)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.06
    )

    # 5. Training
    model.train()

    epochs = 200

    for epoch in range(epochs):
        optimizer.zero_grad()

        logits = model(X_tensor)
        loss = criterion(logits, y_tensor)

        loss.backward()
        optimizer.step()

    # 6. Convert validation data to tensors
    X_val_tensor = torch.tensor(
        np.asarray(X_fold_val_transformed, dtype=np.float32)
    )

    y_val_tensor = torch.tensor(
        y_fold_val.to_numpy(),
        dtype=torch.float32
    ).reshape(-1, 1)

    # 7. Validation
    model.eval()

    with torch.no_grad():
        val_logits = model(X_val_tensor)
        val_loss = criterion(val_logits, y_val_tensor)

        probabilities = torch.sigmoid(val_logits)
        predictions = (probabilities >= 0.5).float()

        accuracy = (predictions == y_val_tensor).float().mean()

    # 8. Save results
    val_losses.append(val_loss.item())
    val_accuracies.append(accuracy.item())

    print(f"Fold {fold}: Loss = {val_loss.item():.4f}, Accuracy = {accuracy.item():.4f}")

# 9. Calculate cross-validation results
print("\nCross-Validation Results")
print(f"Mean Validation Loss: {np.mean(val_losses):.4f}")
print(f"Std Validation Loss: {np.std(val_losses):.4f}")
print(f"Mean Validation Accuracy: {np.mean(val_accuracies):.4f}")

# We implement DR Logistic Regression


def dro_constraint(model, lambda_dual):
    """
    model contains our weight params and lambda_dual contains our optimization variable l
    """
    with torch.no_grad():
        w = model.weight
        a = torch.linalg.vector_norm(w)
        t = lambda_dual.clone()
        if a <= t:
            return
        elif a <= -t:
            w.zero_()
            lambda_dual.zero_()
        else:
            new_lambda = (a + t) / 2
            #Modifica los pesos originales in place
            w.mul_(new_lambda / a)
            lambda_dual.copy_(new_lambda)

# Wasserstein DRO Logistic Regression - Grid Search CV

epsilon_values = [0.001, 0.01, 0.05]
kappa_values = [0.5, 1.0, 2.0]

# Store validation results for every combination
results = {}

for epsilon in epsilon_values:
    for kappa in kappa_values:
        results[(epsilon, kappa)] = {
            "losses": [],
            "accuracies": []
        }

criterion = nn.BCEWithLogitsLoss(reduction="none")

# Cross-Validation
for fold, (train_idx, val_idx) in enumerate(
    cv.split(X_train, y_train), start=1
):

    # 1. Select training and validation data
    X_fold_train = X_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]

    y_fold_train = y_train.iloc[train_idx]
    y_fold_val = y_train.iloc[val_idx]

    # 2. Preprocessing
    X_fold_train_transformed = preprocessor.fit_transform(X_fold_train)
    X_fold_val_transformed = preprocessor.transform(X_fold_val)

    # 3. Convert training data to tensors
    X_tensor = torch.tensor(
        np.asarray(X_fold_train_transformed, dtype=np.float32)
    )

    y_tensor = torch.tensor(
        y_fold_train.to_numpy(),
        dtype=torch.float32
    ).reshape(-1, 1)

    # 4. Convert validation data to tensors
    X_val_tensor = torch.tensor(
        np.asarray(X_fold_val_transformed, dtype=np.float32)
    )

    y_val_tensor = torch.tensor(
        y_fold_val.to_numpy(),
        dtype=torch.float32
    ).reshape(-1, 1)

    # 5. Try every hyperparameter combination
    for epsilon in epsilon_values:
        for kappa in kappa_values:

            # Create a NEW model for each combination
            torch.manual_seed(42 + fold)

            dro_model = nn.Linear(X_tensor.shape[1], 1)

            lambda_var = nn.Parameter(
                torch.linalg.vector_norm(
                    dro_model.weight.detach()
                ).clone()
            )

            # Optimizer
            optimizer = torch.optim.SGD(
                list(dro_model.parameters()) + [lambda_var],
                lr=0.06
            )

            # 6. Training
            dro_model.train()
            epochs = 2000

            for epoch in range(epochs):
                optimizer.zero_grad()

                logits = dro_model(X_tensor)

                loss_real = criterion(logits, y_tensor)
                loss_flipped = criterion(logits, 1 - y_tensor)

                s = torch.maximum(
                    loss_real,
                    loss_flipped - lambda_var * kappa
                )

                dro_loss = lambda_var * epsilon + s.mean()

                dro_loss.backward()
                optimizer.step()

                # Enforce ||beta||_2 <= lambda
                dro_constraint(dro_model, lambda_var)

            # 7. Validation
            dro_model.eval()

            with torch.no_grad():
                val_logits = dro_model(X_val_tensor)

                val_loss = criterion(
                    val_logits, y_val_tensor
                ).mean()

                probabilities = torch.sigmoid(val_logits)
                predictions = (probabilities >= 0.5).float()

                accuracy = (
                    predictions == y_val_tensor
                ).float().mean()

            # 8. Save results for this combination
            results[(epsilon, kappa)]["losses"].append(
                val_loss.item()
            )

            results[(epsilon, kappa)]["accuracies"].append(
                accuracy.item()
            )

            print(
                f"Fold {fold} | "
                f"epsilon={epsilon}, kappa={kappa} | "
                f"Loss={val_loss.item():.4f}, "
                f"Accuracy={accuracy.item():.4f}"
            )

# 9. Print average results for all combinations
print("\nDRO Grid Search Results")

for (epsilon, kappa), metrics in results.items():

    mean_loss = np.mean(metrics["losses"])
    mean_accuracy = np.mean(metrics["accuracies"])

    print(
        f"epsilon={epsilon}, kappa={kappa} | "
        f"Mean Loss={mean_loss:.4f}, "
        f"Mean Accuracy={mean_accuracy:.4f}"
    )

# 10. Select best hyperparameters by Validation BCE
best_params = min(
    results,
    key=lambda params: np.mean(results[params]["losses"])
)

best_epsilon, best_kappa = best_params

print("\nBest Hyperparameters")
print("Best epsilon:", best_epsilon)
print("Best kappa:", best_kappa)
print(
    "Best Mean Validation Loss:",
    np.mean(results[best_params]["losses"])
)

# FINAL TRAINING AND TESTING

# 1. Fit preprocessing on the complete training set
final_preprocessor = clone(preprocessor)

X_train_transformed = final_preprocessor.fit_transform(X_train)

# The test set uses the preprocessing learned from training
X_test_transformed = final_preprocessor.transform(X_test)

# 2. Convert training data to tensors

X_train_tensor = torch.tensor(
    np.asarray(X_train_transformed, dtype=np.float32)
)

y_train_tensor = torch.tensor(
    y_train.to_numpy(),
    dtype=torch.float32
).reshape(-1, 1)

# 3. Convert test data to tensors

X_test_tensor = torch.tensor(
    np.asarray(X_test_transformed, dtype=np.float32)
)

y_test_tensor = torch.tensor(
    y_test.to_numpy(),
    dtype=torch.float32
).reshape(-1, 1)

print("Training shape:", X_train_tensor.shape)
print("Test shape:", X_test_tensor.shape)

# FINAL LOGISTIC REGRESSION

# 1. Create model
torch.manual_seed(42)

lr_model = nn.Linear(X_train_tensor.shape[1], 1)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    lr_model.parameters(),
    lr=0.01
)

# 2. Training
lr_model.train()

epochs = 400

for epoch in range(epochs):
    optimizer.zero_grad()

    logits = lr_model(X_train_tensor)

    loss = criterion(logits, y_train_tensor)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}: LR Loss = {loss.item():.4f}")

# FINAL WASSERSTEIN DRO LOGISTIC REGRESSION

best_epsilon = 0.001
best_kappa = 2.0

# 1. Create model
torch.manual_seed(42)

dro_model = nn.Linear(X_train_tensor.shape[1], 1)

# 2. Create lambda optimization variable
lambda_var = nn.Parameter(
    torch.linalg.vector_norm(
        dro_model.weight.detach()
    ).clone()
)

# 3. Optimizer
optimizer = torch.optim.SGD(
    list(dro_model.parameters()) + [lambda_var],
    lr=0.06
)

# 4. Training
dro_model.train()

epochs = 2000

for epoch in range(epochs):

    optimizer.zero_grad()

    logits = dro_model(X_train_tensor)

    # Logistic loss with original labels
    loss_real = F.binary_cross_entropy_with_logits(
        logits,
        y_train_tensor,
        reduction="none"
    )

    # Logistic loss with inverted labels
    loss_flipped = F.binary_cross_entropy_with_logits(
        logits,
        1 - y_train_tensor,
        reduction="none"
    )

    # Optimal auxiliary variables s_i
    s = torch.maximum(
        loss_real,
        loss_flipped - lambda_var * best_kappa
    )

    # Wasserstein DRO objective
    dro_loss = lambda_var * best_epsilon + s.mean()

    # Backpropagation
    dro_loss.backward()

    # Gradient descent
    optimizer.step()

    # Enforce ||beta||_2 <= lambda
    dro_constraint(dro_model, lambda_var)

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch + 1}: DRO Loss = {dro_loss.item():.4f}")

# 5. Verify the constraint

norm_beta = torch.linalg.vector_norm(dro_model.weight).item()

print("\nFinal DRO Parameters")
print("Lambda:", lambda_var.item())
print("Norm beta:", norm_beta)
print("Constraint satisfied:", norm_beta <= lambda_var.item() + 1e-6)

# FINAL TEST EVALUATION

def evaluate_model(model, X_test_tensor, y_test_tensor, model_name):

    # 1. Evaluation mode
    model.eval()

    # 2. Predict probabilities
    with torch.no_grad():
        logits = model(X_test_tensor)

        probabilities = torch.sigmoid(logits)

        predictions = (probabilities >= 0.5).float()

    # 3. Convert tensors to NumPy
    y_true = y_test_tensor.numpy().ravel().astype(int)
    y_pred = predictions.numpy().ravel().astype(int)
    y_prob = probabilities.numpy().ravel()

    # 4. Calculate metrics
    metrics = {
        "Model": model_name,
        "BCE": log_loss(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob)
    }

    # 5. Print confusion matrix
    print(f"\n{model_name} - Confusion Matrix")
    print(confusion_matrix(y_true, y_pred))

    # 6. Print classification report
    print(f"\n{model_name} - Classification Report")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["Not Canceled", "Canceled"],
        zero_division=0
    ))

    return metrics, y_prob

# COMPARE MODELS ON THE TEST SET

# 1. Evaluate standard Logistic Regression
lr_metrics, lr_probabilities = evaluate_model(
    lr_model,
    X_test_tensor,
    y_test_tensor,
    "Logistic Regression"
)

# 2. Evaluate Wasserstein DRO Logistic Regression
dro_metrics, dro_probabilities = evaluate_model(
    dro_model,
    X_test_tensor,
    y_test_tensor,
    "Wasserstein DRO-LR"
)

# 3. Compare final metrics
results_df = pd.DataFrame([
    lr_metrics,
    dro_metrics
])

print("\nFINAL TEST RESULTS")
print(results_df.round(4).to_string(index=False))

# 4. Majority-class baseline
y_true = y_test_tensor.numpy().ravel()

baseline_accuracy = max(
    np.mean(y_true == 0),
    np.mean(y_true == 1)
)

print(f"\nMajority Baseline Accuracy: {baseline_accuracy:.4f}")

# FINAL EVALUATION PLOTS

y_true = y_test_tensor.numpy().ravel().astype(int)

# 1. ROC Curves
fig, ax = plt.subplots(figsize=(7, 5))

RocCurveDisplay.from_predictions(
    y_true, lr_probabilities,
    name="Logistic Regression",
    ax=ax
)

RocCurveDisplay.from_predictions(
    y_true, dro_probabilities,
    name="Wasserstein DRO-LR",
    ax=ax
)

ax.set_title("ROC Curve - Test Set")
plt.show()

# 2. Precision-Recall Curves
fig, ax = plt.subplots(figsize=(7, 5))

PrecisionRecallDisplay.from_predictions(
    y_true, lr_probabilities,
    name="Logistic Regression",
    ax=ax
)

PrecisionRecallDisplay.from_predictions(
    y_true, dro_probabilities,
    name="Wasserstein DRO-LR",
    ax=ax
)

ax.set_title("Precision-Recall Curve - Test Set")
plt.show()
