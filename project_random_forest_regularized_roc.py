import pandas as pd
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
# Import metrics including roc_curve and auc
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
# Import StandardScaler (might be needed if you decide to scale for RF later)
from sklearn.preprocessing import StandardScaler
import numpy as np
# Import plotting library
import matplotlib.pyplot as plt
# Import warnings filter (optional, can suppress warnings if needed)
import warnings
from sklearn.exceptions import ConvergenceWarning

# Filter potential warnings (less likely for RF, but good practice)
# warnings.filterwarnings("ignore", category=ConvergenceWarning)


# --- 1. Load and Reshape Data ---
# Load your data
try:
    df = pd.read_csv('features2.csv', header=None, names=['sample', 'snp', 'phenotype'])
except FileNotFoundError:
    print("Error: 'features2.csv' not found. Please ensure the file is in the correct directory.")
    exit()
except pd.errors.EmptyDataError:
    print("Error: 'features2.csv' is empty.")
    exit()
except Exception as e:
    print(f"Error loading 'features2.csv': {e}")
    exit()

# Create a binary matrix
df['present'] = 1
data_pivot = df.pivot_table(index='sample', columns='snp', values='present', fill_value=0)

# Get the phenotype for each sample
phenotypes = df.drop_duplicates(subset='sample').set_index('sample')['phenotype']

# Align pivot table index with phenotypes index
phenotypes = phenotypes[phenotypes.index.isin(data_pivot.index)]
data_pivot = data_pivot.loc[phenotypes.index]

# Features (X - original SNP matrix) and target (y - phenotype vector)
X = data_pivot.values
y = phenotypes.values
snp_names = data_pivot.columns # Store SNP names

# Ensure target variable y is suitable for ROC (e.g., 0 and 1)
y_roc = np.where(y == -1, 0, 1) # Map -1 -> 0, 1 -> 1

print(f"Original feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"Total number of samples: {X.shape[0]}")

# --- PCA Section Removed ---
# We will use the original features X directly in the Random Forest.
# Random Forests are generally less sensitive to feature scaling than linear models.
# We will use X (unscaled) as input. If performance is poor, consider scaling X.
X_processed = X
input_feature_desc = "original features"
print(f"\nUsing {input_feature_desc} for Random Forest.")


# --- 3. Regularized Random Forest with Leave-One-Out Cross-Validation (LOOCV) ---

# Initialize the Random Forest Classifier with regularization parameters
# These help prevent overfitting, especially with many original features.
# max_depth: Limits tree depth.
# min_samples_leaf: Requires a minimum number of samples in a leaf node.
# max_features: Limits the number of features considered at each split (sqrt is common).
# ccp_alpha: Cost-complexity pruning parameter.
# Tune these values for your specific dataset.
rf_classifier = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    random_state=0,          # For reproducibility
    max_depth=10,            # Regularization: Limit tree depth
    min_samples_leaf=2,      # Regularization: Require samples per leaf
    max_features='sqrt',     # Regularization: Limit features per split
    # ccp_alpha=0.01         # Regularization: Optional pruning parameter
)

loo = LeaveOneOut()
y_true_list = []
y_pred_list = []
y_scores_list = [] # List to store predicted probabilities for the positive class

print(f"\nStarting Leave-One-Out Cross-Validation (using Regularized Random Forest on {input_feature_desc})...")

# Perform LOOCV using the original features (X_processed = X)
for i, (train_index, test_index) in enumerate(loo.split(X_processed)):
    X_train, X_test = X_processed[train_index], X_processed[test_index]
    # Use original y for training (-1, 1)
    y_train, y_test = y[train_index], y[test_index]

    # Train the Random Forest model
    rf_classifier.fit(X_train, y_train)

    # Predict class label
    y_pred = rf_classifier.predict(X_test)

    # Predict probabilities
    y_proba = rf_classifier.predict_proba(X_test)

    # Store the true label (original) and the prediction
    y_true_list.append(y_test[0]) # Original labels (-1, 1)
    y_pred_list.append(y_pred[0]) # Predicted labels (-1, 1)

    # Find index corresponding to the positive class (1)
    try:
        positive_class_index = np.where(rf_classifier.classes_ == 1)[0][0]
        y_scores_list.append(y_proba[0, positive_class_index])
    except IndexError:
        print(f"Warning: Could not find class 1 in fold {i}. Classifier classes: {rf_classifier.classes_}. Probabilities: {y_proba}. Appending 0.5 score.")
        y_scores_list.append(0.5) # Append a neutral probability


# Convert lists to numpy arrays for evaluation
y_true_final = np.array(y_true_list)
y_pred_final = np.array(y_pred_list)
y_scores_final = np.array(y_scores_list)
# Use the ROC-compatible true labels (0, 1) for ROC calculation
y_true_roc_final = np.where(y_true_final == -1, 0, 1)


# --- 4. Evaluate Model Performance ---
accuracy = accuracy_score(y_true_final, y_pred_final)
# Use original labels (-1, 1) for confusion matrix and classification report
conf_matrix = confusion_matrix(y_true_final, y_pred_final, labels=[-1, 1])
class_report = classification_report(y_true_final, y_pred_final, target_names=['Control (-1)', 'Case (1)'], labels=[-1, 1], zero_division=0)

# Calculate ROC curve and AUC using the probabilities
fpr, tpr, thresholds = roc_curve(y_true_roc_final, y_scores_final, pos_label=1)
roc_auc = auc(fpr, tpr)

print("\n--- LOOCV Results (Regularized Random Forest on Original Features) ---")
print(f"**Overall Accuracy**: {accuracy:.4f}")

print("\n**Confusion Matrix**:")
print("Predicted:  Control (-1)  Case (1)")
print(f"True Control (-1): {conf_matrix[0,0]:^12} {conf_matrix[0,1]:^10}")
print(f"True Case    ( 1): {conf_matrix[1,0]:^12} {conf_matrix[1,1]:^10}")

print("\n**Classification Report**:")
print(class_report)

# Print the AUC score
print(f"\n**Area Under ROC Curve (AUC)**: {roc_auc:.4f}")


# --- 5. Plot ROC Curve (Optional) ---
print("\nGenerating ROC Curve plot...")
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title(f'Receiver Operating Characteristic (ROC) Curve - LOOCV (Regularized RF on {input_feature_desc})')
plt.legend(loc="lower right")
plt.grid(True)
# plt.savefig('roc_curve_rf_regularized_original_features.png')
plt.show()

# --- 6. Feature Importance (Optional - based on final model) ---
# Train final model on ALL original data to see feature importances
print("\n--- Feature Importances (from final model trained on all data) ---")

# Train final regularized RF model on the full original dataset X
rf_final = RandomForestClassifier(
    n_estimators=100, random_state=0, max_depth=10,
    min_samples_leaf=3, max_features='sqrt' # Use same regularization
)
rf_final.fit(X, y) # Train on original X and y

# Get feature importances (mean decrease in impurity)
importances = rf_final.feature_importances_
indices = np.argsort(importances)[::-1] # Sort features by importance

# Print the feature ranking
print("Feature ranking (Top 20):")
n_top_features = min(20, X.shape[1]) # Show top 20 or fewer if not available
for f in range(n_top_features):
    feature_index = indices[f]
    print(f"{f + 1}. Feature '{snp_names[feature_index]}' ({importances[feature_index]:.4f})")


print("\nScript finished.")
