import pandas as pd
from io import StringIO
# Import LogisticRegression for L1 (Lasso) penalty
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
# Import metrics including roc_curve and auc
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
# Import StandardScaler for scaling within CV loop
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import numpy as np
# Import plotting library
import matplotlib.pyplot as plt
# Import warnings filter to suppress convergence warnings if they occur
import warnings
from sklearn.exceptions import ConvergenceWarning

# Filter convergence warnings from Logistic Regression (can happen with small N/strong penalty)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


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

# Features (X) and target (y)
X = data_pivot.values
y = phenotypes.values
snp_names = data_pivot.columns # Store SNP names

# Ensure target variable y is suitable for ROC (e.g., 0 and 1)
y_roc = np.where(y == -1, 0, 1) # Map -1 -> 0, 1 -> 1

print(f"Original feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"Total number of samples: {X.shape[0]}")

# --- 2. Apply UMAP for Dimensionality Reduction ---
n_samples = X.shape[0]
umap_applied = False
X_processed = X # Default to original features if UMAP fails/skipped
umap_reducer = None

if n_samples <= 1:
    print("Error: Not enough samples to perform analysis.")
    exit()

# Adjust UMAP parameters
n_neighbors_umap = min(10, n_samples - 1)
if n_neighbors_umap <= 1:
     print(f"Warning: n_neighbors ({n_neighbors_umap}) is very small. UMAP might not be effective. Skipping UMAP.")
     X_processed = X # Use original features
else:
    # Choose the number of components
    n_components_umap = min(n_samples - 1, 5)
    print(f"\nApplying UMAP: n_neighbors={n_neighbors_umap}, n_components={n_components_umap}")

    umap_reducer = UMAP(n_neighbors=n_neighbors_umap,
                        n_components=n_components_umap,
                        random_state=0,
                        min_dist=0.1)
    try:
        X_reduced = umap_reducer.fit_transform(X)
        X_processed = X_reduced # Use UMAP-reduced features for the model
        umap_applied = True
        print(f"Reduced feature matrix shape (UMAP): {X_processed.shape}")
    except Exception as e:
        print(f"Error during UMAP transformation: {e}")
        print("Using original features instead of UMAP.")
        X_processed = X
        umap_reducer = None


# --- 3. Logistic Regression (Lasso) with Leave-One-Out Cross-Validation (LOOCV) ---
# Initialize the Logistic Regression Classifier with L1 penalty
# C is the inverse of regularization strength; smaller C = stronger regularization
# 'liblinear' solver is good for L1 penalty. 'saga' is another option.
lasso_classifier = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=0)

loo = LeaveOneOut()
y_true_list = []
y_pred_list = []
y_scores_list = [] # List to store predicted probabilities for the positive class

print("\nStarting Leave-One-Out Cross-Validation (using Logistic Regression with L1 penalty on " + ("UMAP features" if umap_applied else "original features") + ")...")

# Perform LOOCV
for i, (train_index, test_index) in enumerate(loo.split(X_processed)):
    X_train_orig, X_test_orig = X_processed[train_index], X_processed[test_index]
    # Use original y for training (-1, 1)
    y_train, y_test = y[train_index], y[test_index]

    # --- Scaling within the loop ---
    # Initialize scaler IN THE LOOP
    scaler_cv = StandardScaler()
    # Fit scaler ONLY on the training data for this fold
    X_train_scaled = scaler_cv.fit_transform(X_train_orig)
    # Transform the test data using the scaler fitted on the training data
    X_test_scaled = scaler_cv.transform(X_test_orig)
    # --- End Scaling ---

    # Train the Lasso classifier on the SCALED training data
    lasso_classifier.fit(X_train_scaled, y_train)

    # Predict class label on the SCALED test data
    y_pred = lasso_classifier.predict(X_test_scaled)

    # Predict probabilities on the SCALED test data
    y_proba = lasso_classifier.predict_proba(X_test_scaled)

    # Store the true label (original) and the prediction
    y_true_list.append(y_test[0]) # Original labels (-1, 1) for confusion matrix/report
    y_pred_list.append(y_pred[0]) # Predicted labels (-1, 1)

    # Find index corresponding to the positive class (1) in the classifier's classes_
    try:
        positive_class_index = np.where(lasso_classifier.classes_ == 1)[0][0]
        # Store the probability of the positive class (1)
        y_scores_list.append(y_proba[0, positive_class_index])
    except IndexError:
        # Handle cases where a class might not be predicted (unlikely in LOOCV but possible)
        # Or if classes_ doesn't contain 1 for some reason (e.g., only one class in training data - should not happen in LOOCV)
        print(f"Warning: Could not find class 1 in fold {i}. Classifier classes: {lasso_classifier.classes_}. Probabilities: {y_proba}. Appending 0.5 score.")
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
# Use y_true_roc_final (0, 1) and y_scores_final (probabilities of class 1)
fpr, tpr, thresholds = roc_curve(y_true_roc_final, y_scores_final, pos_label=1)
roc_auc = auc(fpr, tpr)

print("\n--- LOOCV Results (Logistic Regression L1) ---")
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
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonal line for random classifier
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve - LOOCV (Lasso)')
plt.legend(loc="lower right")
plt.grid(True)
# You can save the plot instead of showing it:
# plt.savefig('roc_curve_umap_lasso.png')
plt.show() # Display the plot

# --- 6. Feature Importance (Optional - based on final model) ---
# Train final model on ALL processed & scaled data to see coefficients
if umap_applied or X_processed is not X: # Check if any processing was done
    print("\n--- Feature Coefficients (from final model trained on all processed data) ---")
    # Scale the full processed dataset
    scaler_final = StandardScaler()
    X_processed_scaled = scaler_final.fit_transform(X_processed)

    # Train final Lasso model
    lasso_final = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=0)
    lasso_final.fit(X_processed_scaled, y)

    # Get coefficients (these relate to the UMAP components or original features if UMAP was skipped)
    coefficients = lasso_final.coef_[0] # coef_ is shape (1, n_features) for binary classification

    if umap_applied:
        feature_names = [f"UMAP_{i+1}" for i in range(X_processed_scaled.shape[1])]
    else:
        # If UMAP was skipped, X_processed is X, but we should use snp_names
        # However, the model was trained on X_processed_scaled, which corresponds to X
        # Ensure snp_names length matches X_processed_scaled.shape[1]
        if X_processed_scaled.shape[1] == len(snp_names):
             feature_names = snp_names
        else:
             feature_names = [f"Feature_{i+1}" for i in range(X_processed_scaled.shape[1])]


    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    # Filter out features with zero coefficient (due to Lasso)
    coef_df = coef_df[np.abs(coef_df['Coefficient']) > 1e-6] # Use small threshold for floating point
    coef_df = coef_df.sort_values(by='Coefficient', key=abs, ascending=False)

    print("Non-zero coefficients (indicating feature importance by Lasso):")
    print(coef_df.to_string(index=False))

else:
    print("\nSkipping final coefficient analysis as original features were used without scaling/processing.")


print("\nScript finished.")
