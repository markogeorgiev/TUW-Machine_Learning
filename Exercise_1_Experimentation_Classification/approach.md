# General Guide: ML Classification Exercise

Based on the exercise description and selected classifiers (k-NN, Random Forest, MLP), here is a structured guide on how to complete the task, including what versions to implement and how to compare models.

---

## 1. Classifier Versions to Implement

You are expected to experiment with different settings for each classifier. Implement and compare the following versions:

### k-NN
- Vary `k` values: 1, 3, 5, 7, 9
- Try different distance metrics: Euclidean (default), Manhattan
- Compare performance with and without feature scaling

### Random Forest Classifier (RFC)
- Vary `n_estimators`: 10, 50, 100, 200
- Vary `max_depth`: None, 5, 10, 20
- Try different criteria: `gini`, `entropy`
- Optionally try with `bootstrap=True` or `False`

### MLP
- Vary hidden layer architectures: `[64]`, `[128, 64]`, `[128, 64, 32]`
- Vary learning rates: 0.01, 0.001, 0.0001
- Try with and without dropout
- Train with different epoch counts: 30, 50, 100

---

## 2. How to Compare Models

### a. Within Each Classifier
- Track performance changes when hyperparameters are varied
- Use cross-validation (e.g., 5-fold) for fair evaluation
- Create comparison tables or line plots
  - x-axis: hyperparameter (e.g., k, n_estimators)
  - y-axis: accuracy or F1-score

### b. Between Classifiers
For each dataset:
- Report best configuration for each model
- Use consistent metrics:
  - Accuracy, Precision, Recall, F1-score, (optional: AUC)
- Visualize results with bar charts or tables
- Discuss:
  - Accuracy and generalization
  - Runtime and scalability
  - Sensitivity to preprocessing

---

## 3. Experimental Design per Dataset

For each of your 4 datasets:
1. Describe dataset: size, target, class distribution, etc.
2. Preprocessing:
   - Handle missing values
   - Encode categorical features
   - Scale features if needed (especially for MLP and k-NN)
3. Train all classifiers using various parameter settings
4. Evaluate using cross-validation
5. Aggregate results into tables and visualizations

---

## 4. Metrics to Use

Include at least:
- Accuracy
- Precision, Recall, F1-score (especially for imbalanced data)
- Optional: AUC-ROC for binary classification
- Optional: Training time for efficiency

---

## 5. Result Presentation

- Present per-dataset tables comparing all classifiers and settings
- Include one final summary table with best results across all datasets and models
- Discuss:
  - Best classifier per dataset
  - Notable trends or unexpected results
  - Which models were most sensitive to preprocessing or hyperparameters
  - Effect of dataset size and dimensionality

---

## 6. Kaggle Submission Tips

- Choose two Kaggle datasets
- Format submission: `<id>,<predicted class>`
- Compare your performance to the leaderboard for feedback

---

## 7. Report Writing Tips

- Use well-organized tables and readable plots
- Clearly justify preprocessing and parameter choices
- Discuss classifier strengths/weaknesses per dataset
- Compare cross-validation vs holdout results
- Provide insightful analysis, not just results

---

Would recommend tracking all results using a structured DataFrame or spreadsheet and visualizing trends for better comparison.
