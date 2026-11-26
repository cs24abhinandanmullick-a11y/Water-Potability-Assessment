
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.pipeline import Pipeline


# =====================================
# STEP 1: LOAD DATASET
# =====================================

df = pd.read_csv("water_potability.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst five records:")
print(df.head())


# =====================================
# STEP 2: DATA EXPLORATION
# =====================================

print("\nMissing values:")
print(df.isnull().sum())

print("\nOriginal target distribution:")
print(df["Potability"].value_counts())


# =====================================
# STEP 3: SEPARATE X AND Y
# =====================================

X = df.drop("Potability", axis=1)
y = df["Potability"]


# =====================================
# STEP 4: TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# =====================================
# STEP 5: CREATE DECISION TREE PIPELINE
# =====================================

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),

    ("classifier", DecisionTreeClassifier(
        random_state=42
    ))
])


# =====================================
# STEP 6: HYPERPARAMETER TUNING
# =====================================

param_grid = {

    "classifier__criterion": [
        "gini",
        "entropy"
    ],

    "classifier__max_depth": [
        3, 5, 7, 10, 15, None
    ],

    "classifier__min_samples_split": [
        2, 5, 10, 20
    ],

    "classifier__min_samples_leaf": [
        1, 2, 5, 10
    ]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

model = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    verbose=1
)


# =====================================
# STEP 7: TRAIN OPTIMIZED MODEL
# =====================================

print("\nStarting hyperparameter optimization...")

model.fit(X_train, y_train)

print("\nModel training completed!")

print("\nBEST DECISION TREE PARAMETERS:")
print(model.best_params_)

print("\nBEST CROSS-VALIDATION ACCURACY:")
print(round(model.best_score_ * 100, 2), "%")


# =====================================
# STEP 8: MAKE PREDICTIONS
# =====================================

y_pred = model.predict(X_test)

train_pred = model.predict(X_train)


# =====================================
# STEP 9: MODEL EVALUATION
# =====================================

print("\nTRAINING ACCURACY:",
      round(accuracy_score(y_train, train_pred) * 100, 2), "%")

print("\nTESTING ACCURACY:",
      round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nPRECISION:",
      round(precision_score(y_test, y_pred, zero_division=0) * 100, 2), "%")

print("\nRECALL:",
      round(recall_score(y_test, y_pred, zero_division=0) * 100, 2), "%")

print("\nF1 SCORE:",
      round(f1_score(y_test, y_pred, zero_division=0) * 100, 2), "%")


print("\nCLASSIFICATION REPORT:")

print(classification_report(
    y_test,
    y_pred,
    target_names=["Not Potable", "Potable"],
    zero_division=0
))


# =====================================
# STEP 10: CONFUSION MATRIX
# =====================================

cm = confusion_matrix(y_test, y_pred)

print("\nCONFUSION MATRIX:")
print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Potable", "Potable"],
    yticklabels=["Not Potable", "Potable"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Water Potability Confusion Matrix")

plt.tight_layout()
plt.show()


# =====================================
# STEP 11: DECISION TREE DISPLAY
# =====================================

best_pipeline = model.best_estimator_

tree_model = best_pipeline.named_steps["classifier"]

plt.figure(figsize=(24, 12))

plot_tree(
    tree_model,
    feature_names=X.columns,
    class_names=["Not Potable", "Potable"],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("Optimized Water Potability Decision Tree")

plt.tight_layout()
plt.show()


# =====================================
# STEP 12: FEATURE IMPORTANCE
# =====================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": tree_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE:")
print(importance.to_string(index=False))


# =====================================
# STEP 13: USER INPUT WITH UNITS
# =====================================

print("\n=====================================")
print("       WATER POTABILITY TEST")
print("=====================================")

print("\nEnter the water sample parameters:")
print("(Please enter values in the specified units)\n")

ph = float(input("Enter pH (pH units): "))

hardness = float(input("Enter Hardness (mg/L as CaCO3): "))

solids = float(input("Enter Solids (ppm): "))

chloramines = float(input("Enter Chloramines (mg/L): "))

sulfate = float(input("Enter Sulfate (mg/L): "))

conductivity = float(input("Enter Conductivity (µS/cm): "))

organic_carbon = float(input("Enter Organic Carbon (mg/L): "))

trihalomethanes = float(input("Enter Trihalomethanes (µg/L): "))

turbidity = float(input("Enter Turbidity (NTU): "))


# =====================================
# STEP 14: CREATE USER SAMPLE
# =====================================

sample = pd.DataFrame([{
    "ph": ph,
    "Hardness": hardness,
    "Solids": solids,
    "Chloramines": chloramines,
    "Sulfate": sulfate,
    "Conductivity": conductivity,
    "Organic_carbon": organic_carbon,
    "Trihalomethanes": trihalomethanes,
    "Turbidity": turbidity
}])


# =====================================
# STEP 15: PREDICT WATER POTABILITY
# =====================================

prediction = model.predict(sample)

print("\n=====================================")
print("          PREDICTION RESULT")
print("=====================================")

if prediction[0] == 1:

    print("RESULT: WATER IS PREDICTED TO BE POTABLE")

else:

    print("RESULT: WATER IS PREDICTED TO BE NOT POTABLE")


# =====================================
# STEP 16: PREDICTION PROBABILITY
# =====================================

probability = model.predict_proba(sample)

print("\nPREDICTION PROBABILITIES:")

print("Not Potable:",
      round(probability[0][0] * 100, 2), "%")

print("Potable:",
      round(probability[0][1] * 100, 2), "%")


print("\n=====================================")
print("       TESTING COMPLETED")
print("=====================================")
