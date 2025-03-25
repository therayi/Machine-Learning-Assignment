# Machine-Learning-Assignment
# Feature Importance & Model Interpretation in Machine Learning

## Overview
This repository contains a tutorial on **Feature Importance & Model Interpretation**, covering weight-based importance (e.g., XGBoost), permutation importance, and partial dependence plots (PDPs). The tutorial explains these techniques using Python with real-world datasets and visualization examples.

## Implementation Steps

### **1. Install Dependencies**
Before running the code, ensure that all required Python libraries are installed:
```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```

### **2. Load the Dataset**
The dataset used in this tutorial is the California housing dataset from Scikit-learn.
```python
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target
```
![image](https://github.com/user-attachments/assets/f02455b0-273a-468f-8fbd-d344a2b81189)

### **3. Train a Machine Learning Model (XGBoost)**
We train an XGBoost model to demonstrate weight-based feature importance.
```python
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

X = df.drop(columns=['Target'])
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor()
model.fit(X_train, y_train)
```

### **4. Feature Importance (Weight-based)**
Visualizing the feature importance based on model weights.
```python
import matplotlib.pyplot as plt
from xgboost import plot_importance

plot_importance(model, importance_type='weight')
plt.title("Feature Importance (Weight-based)")
plt.show()
```
![image](https://github.com/user-attachments/assets/eb2daafc-ced9-4ad2-9a9c-5fb558849f58)

### **5. Permutation Importance**
Using permutation importance to evaluate feature significance.
```python
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(model, X_test, y_test)

sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(X_test.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Importance")
plt.title("Permutation Importance of Features")
plt.show()
```
![image](https://github.com/user-attachments/assets/eead4155-ba09-4382-8839-e8e643bcaf8d)

### **6. Partial Dependence Plots (PDPs)**
Generating PDPs for key features.
```python
from sklearn.inspection import partial_dependence, PartialDependenceDisplay

features = ['MedInc', 'Latitude', 'Longitude']
PartialDependenceDisplay.from_estimator(model, X_test, features)
plt.show()
```
![image](https://github.com/user-attachments/assets/1a1629a8-6f25-47f8-ad89-fde22dba145f)

## References

1. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. [Paper](https://arxiv.org/abs/1603.02754)
2. Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*. [Paper](https://arxiv.org/abs/1705.07874)
3. Breiman, L. (2001). *Random Forests*. [Paper](https://link.springer.com/article/10.1023/A:1010933404324)
4. Molnar, C. (2022). *Interpretable Machine Learning: A Guide*. [Book](https://christophm.github.io/interpretable-ml-book/)
5. Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. [Paper](https://projecteuclid.org/euclid.aos/1013203451)
6. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. [Paper](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)

---

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

