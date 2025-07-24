# 🌸 Iris Species Classification using XGBoost

This project demonstrates a complete machine learning pipeline for classifying iris flowers into three species using the famous Iris dataset. The pipeline includes data preprocessing, visualization, outlier removal, and training an XGBoost classifier.

---

## 📂 Dataset

The dataset used is a modified version of the classic Iris dataset which includes some missing values (`NaN`). The features are:

- `SepalLengthCm`
- `SepalWidthCm`
- `PetalLengthCm`
- `PetalWidthCm`
- `Species` (Target variable)

---

## 🛠️ Preprocessing Steps

1. **Read CSV File**  
   Loaded from `datasets/data_with_nans.csv`

2. **Drop Unnecessary Columns**  
   Dropped `Unnamed: 0` and `Id`

3. **Handle Missing Values**  
   Filled using **class-wise mean imputation**

4. **Outlier Detection and Removal**  
   - Using **3-Sigma Rule**
   - Using **IQR (Interquartile Range) Method**

5. **Label Encoding**  
   Encoded `Species` using `LabelEncoder`

6. **Train-Test Split**  
   80% training, 20% testing

---

## 📊 Visualization

Scatter plots were created for each feature colored by species:
- Before and after removing missing values
- Before and after outlier removal

---

## 🤖 Model: XGBoost Classifier

- Objective: `"multi:softmax"`
- Classes: `3` (Setosa, Versicolor, Virginica)

### 🔧 Training and Evaluation

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


⸻

🧾 Final Output
	•	Cleaned and preprocessed dataset saved as: final_data.csv

⸻

📁 Repository Structure

├── datasets/
│   └── data_with_nans.csv
├── final_data.csv
├── iris_classification.ipynb
├── README.md


⸻

📌 Requirements
	•	Python 3.7+
	•	pandas
	•	seaborn
	•	matplotlib
	•	scikit-learn
	•	xgboost
