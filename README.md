# 🌸 Iris Species Classification using XGBoost

This project demonstrates a complete machine learning pipeline for classifying iris flowers into three species using the famous Iris dataset. The pipeline includes data preprocessing, visualization, outlier removal, and training an XGBoost classifier.

## 📂 Dataset

The dataset used is a modified version of the classic Iris dataset which includes some missing values (`NaN`). The features are:

- `SepalLengthCm`
- `SepalWidthCm`
- `PetalLengthCm`
- `PetalWidthCm`
- `Species` (Target variable)

## 🛠️ Preprocessing Steps

1. **Read CSV File**
   - Dataset loaded from `datasets/data_with_nans.csv`

2. **Drop Unnecessary Columns**
   - Dropped `Unnamed: 0` and `Id` columns

3. **Handle Missing Values**
   - Missing values are filled with **class-wise mean imputation**

4. **Outlier Detection and Removal**
   - **3-Sigma Rule**
   - **IQR (Interquartile Range) Method**

5. **Label Encoding**
   - `Species` is encoded into numerical labels using `LabelEncoder`

6. **Train-Test Split**
   - 80% training, 20% testing with `random_state=42`

## 📊 Visualization

Scatter plots of each feature colored by species were created:
- Before and after removing missing values
- Before and after outlier removal

## 🤖 Model: XGBoost Classifier

- `objective="multi:softmax"`
- `num_class=3`

### Training and Evaluation

```python
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

model = XGBClassifier(objective="multi:softmax", num_class=3)
model.fit(X_train, y_train)
preds = model.predict(X_test)


✅ Results
	•	Accuracy: 94%
	•	Confusion Matrix:

[[11  0  0]
 [ 0 10  0]
 [ 0  0 10]]

🧾 Final Output
	•	Cleaned and preprocessed data is saved as final_data.csv

📁 Repository Structure

├── datasets/
│   └── data_with_nans.csv
├── final_data.csv
├── iris_classification.ipynb
├── README.md

📌 Requirements
	•	Python 3.7+
	•	pandas
	•	seaborn
	•	matplotlib
	•	scikit-learn
	•	xgboost

Install requirements:

pip install pandas seaborn matplotlib scikit-learn xgboost

📬 Contact

If you have questions or suggestions, feel free to reach out via issues or fork this repo and contribute!
```

⸻

Author: Fazlı Altun
License: MIT

---

Let me know if you'd like me to include a Jupyter badge (e.g., “Open in Colab”), or change the tone (academic, beginner-friendly, etc.).
