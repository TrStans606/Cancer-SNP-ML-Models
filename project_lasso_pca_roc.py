import pandas as pd
from io import StringIO
# Import LogisticRegression for L1 (Lasso) penalty
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
# Import metrics including roc_curve and auc
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
# Import PCA and StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
# Import plotting library
import matplotlib.pyplot as plt
# Import warnings filter to suppress convergence warnings if they occur
import warnings
from sklearn.exceptions import ConvergenceWarning

# Filter convergence warnings from Logistic Regression
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

# --- 2. Apply PCA for Dimensionality Reduction ---
n_samples = X.shape[0]
n_features = X.shape[1]
pca_applied = False
X_processed = X # Default to original features if PCA fails/skipped
pca_reducer = None
scaler_pca = None # Keep track of the scaler used before PCA

if n_samples <= 1:
    print("Error: Not enough samples to perform analysis.")
    exit()

# PCA requires features to be scaled (mean=0, variance=1)
print("\nScaling data for PCA...")
scaler_pca = StandardScaler() # Use a specific name for the PCA scaler
X_scaled = scaler_pca.fit_transform(X)

# Determine the number of components for PCA
max_pca_components = min(n_samples -1, n_features)
if max_pca_components <= 0:
    print("Warning: Not enough samples or features to perform PCA effectively. Skipping PCA.")
    # If skipping PCA, the input to LOOCV should still be scaled for Lasso
    print("Using SCALED original features for Lasso.")
    X_processed = X_scaled # Use scaled original features
else:
    n_components_pca = min(5, max_pca_components)
    print(f"Applying PCA: n_components={n_components_pca}")

    pca_reducer = PCA(n_components=n_components_pca, random_state=0)
    try:
        # Fit PCA on the SCALED data and transform it
        X_reduced = pca_reducer.fit_transform(X_scaled)
        # The input to LOOCV is now the PCA-reduced data (already based on scaled data)
        X_processed = X_reduced
        pca_applied = True
        print(f"Reduced feature matrix shape (PCA): {X_processed.shape}")
        print(f"Explained variance ratio by component: {pca_reducer.explained_variance_ratio_}")
        print(f"Total explained variance by {n_components_pca} components: {np.sum(pca_reducer.explained_variance_ratio_):.4f}")
    except Exception as e:
        print(f"Error during PCA transformation: {e}")
        print("Using SCALED original features instead of PCA.")
        X_processed = X_scaled # Fallback to scaled original features
        pca_reducer = None


# --- 2.5 Interpret PCA Components (if PCA was applied) ---
# This interpretation remains the same, based on the PCA fit
if pca_applied and pca_reducer is not None:
    print("\n--- Interpreting PCA Components (Loadings) ---")
    n_top_snps_per_component = 5
    loadings = pca_reducer.components_
    for i in range(loadings.shape[0]):
        component_loadings = loadings[i, :]
        sorted_indices = np.argsort(np.abs(component_loadings))[::-1]
        print(f"\n**Top {n_top_snps_per_component} SNPs contributing to Principal Component {i+1}:**")
        for k in range(min(n_top_snps_per_component, len(snp_names))):
            snp_index = sorted_indices[k]
            snp_name = snp_names[snp_index]
            loading_value = component_loadings[snp_index]
            print(f"  - {snp_name}: Loading = {loading_value:.4f}")


# --- 3. Logistic Regression (Lasso) with Leave-One-Out Cross-Validation (LOOCV) ---
# Initialize the Logistic Regression Classifier with L1 penalty
lasso_classifier = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=0)

loo = LeaveOneOut()
y_true_list = []
y_pred_list = []
y_scores_list = [] # List to store predicted probabilities for the positive class

# Determine the input features description for print statement
input_feature_desc = "PCA features" if pca_applied else "SCALED original features"
print(f"\nStarting Leave-One-Out Cross-Validation (using Logistic Regression with L1 penalty on {input_feature_desc})...")

# Perform LOOCV
for i, (train_index, test_index) in enumerate(loo.split(X_processed)):
    # X_processed is either PCA-reduced data (derived from scaled data)
    # or scaled original data (if PCA was skipped/failed)
    X_train, X_test = X_processed[train_index], X_processed[test_index]
    # Use original y for training (-1, 1)
    y_train, y_test = y[train_index], y[test_index]

    # --- Scaling within the loop ---
    # Although PCA input was scaled, scaling again within the loop
    # using only train data ensures robustness, especially if PCA was skipped.
    # If PCA was applied, the features (PCs) are already somewhat standardized,
    # but scaling again doesn't hurt and maintains consistency.
    scaler_cv = StandardScaler()
    X_train_scaled = scaler_cv.fit_transform(X_train)
    X_test_scaled = scaler_cv.transform(X_test)
    # --- End Scaling ---

    # Train the Lasso classifier on the SCALED training data for this fold
    lasso_classifier.fit(X_train_scaled, y_train)

    # Predict class label on the SCALED test data
    y_pred = lasso_classifier.predict(X_test_scaled)

    # Predict probabilities on the SCALED test data
    y_proba = lasso_classifier.predict_proba(X_test_scaled)

    # Store the true label (original) and the prediction
    y_true_list.append(y_test[0]) # Original labels (-1, 1)
    y_pred_list.append(y_pred[0]) # Predicted labels (-1, 1)

    # Find index corresponding to the positive class (1)
    try:
        positive_class_index = np.where(lasso_classifier.classes_ == 1)[0][0]
        y_scores_list.append(y_proba[0, positive_class_index])
    except IndexError:
        print(f"Warning: Could not find class 1 in fold {i}. Classifier classes: {lasso_classifier.classes_}. Probabilities: {y_proba}. Appending 0.5 score.")
        y_scores_list.append(0.5)


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
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title(f'Receiver Operating Characteristic (ROC) Curve - LOOCV (Lasso on {input_feature_desc})')
plt.legend(loc="lower right")
plt.grid(True)
# plt.savefig('roc_curve_pca_lasso.png')
plt.show()

# --- 6. Feature Importance (Optional - based on final model) ---
# Train final model on ALL processed data (scaled appropriately) to see coefficients
print("\n--- Feature Coefficients (from final model trained on all processed data) ---")

# Scale the full processed dataset (either PCA components or scaled original features)
scaler_final = StandardScaler()
X_processed_scaled = scaler_final.fit_transform(X_processed)

# Train final Lasso model
lasso_final = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=0)
lasso_final.fit(X_processed_scaled, y)

# Get coefficients
coefficients = lasso_final.coef_[0]

# Determine feature names based on whether PCA was applied
if pca_applied:
    feature_names = [f"PC_{i+1}" for i in range(X_processed_scaled.shape[1])]
else:
    # If PCA was skipped, X_processed is scaled original features
    if X_processed_scaled.shape[1] == len(snp_names):
         feature_names = snp_names
    else: # Fallback if dimensions don't match somehow
         feature_names = [f"Feature_{i+1}" for i in range(X_processed_scaled.shape[1])]


coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
# Filter out features with essentially zero coefficient
coef_df = coef_df[np.abs(coef_df['Coefficient']) > 1e-6]
coef_df = coef_df.sort_values(by='Coefficient', key=abs, ascending=False)

print(f"Non-zero coefficients for {input_feature_desc} (indicating importance by Lasso):")
if coef_df.empty:
    print("Lasso model resulted in all zero coefficients (strong regularization or no signal).")
else:
    print(coef_df.to_string(index=False))


print("\nScript finished.")
