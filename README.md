# 🧠 Diabetes Prediction with Support Vector Machine

This project builds a machine learning model to predict whether a person is diabetic based on diagnostic health measurements. It uses the **Pima Indians Diabetes dataset** and a **Support Vector Machine (SVM)** classifier with a linear kernel. The dataset is standardized before model training to improve performance.

---

## 📁 Project Files

* `diabetes.csv` – Dataset used for training and evaluation.
* `Diabetes predic.ipynb` – Jupyter notebook containing code for:

  * Exploratory Data Analysis (EDA)
  * Data preprocessing (standardization)
  * Model training (SVM)
  * Prediction on new input data
* `README.md` – Project documentation.

---

## 🧪 Technologies Used

* **Python**
* **Jupyter Notebook**
* **Pandas**, **NumPy** – Data handling
* **scikit-learn** – Preprocessing, model building, and evaluation
* **SVM (Support Vector Classifier)**

---

## 📊 Dataset Features

Each row in the dataset represents diagnostic measurements for a female patient:

| Feature                  | Description                                                           |
| ------------------------ | --------------------------------------------------------------------- |
| Pregnancies              | Number of times pregnant                                              |
| Glucose                  | Plasma glucose concentration                                          |
| BloodPressure            | Diastolic blood pressure (mm Hg)                                      |
| SkinThickness            | Triceps skin fold thickness (mm)                                      |
| Insulin                  | 2-Hour serum insulin (mu U/ml)                                        |
| BMI                      | Body mass index (weight in kg/(height in m)^2)                        |
| DiabetesPedigreeFunction | A function that scores likelihood of diabetes based on family history |
| Age                      | Age in years                                                          |
| Outcome                  | Class variable (0 = non-diabetic, 1 = diabetic)                       |

---

## 🔁 Workflow Overview

### 1. Data Loading & Exploration

* Load `diabetes.csv` using Pandas.
* Check for missing values and descriptive statistics.

### 2. Data Preprocessing

* Drop the **Outcome** column to separate features (X) and target (Y).
* Standardize feature values using `StandardScaler`.

### 3. Train/Test Split

* Use `train_test_split` to divide the dataset (80% training, 20% testing).

### 4. Model Training

* Train an SVM model with a linear kernel on the standardized training data.

### 5. Model Evaluation

* Compute training and testing accuracy.
* Model achieves \~**78.7% accuracy** on training data and **77.3% on testing data**.

### 6. Making Predictions

* Predict using new input values after applying the same standardization.
* Example:

  ```python
  input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)
  prediction = classifier.predict(scaled_input)
  ```

---

## 🚀 Getting Started

### ✅ Prerequisites

Install the required Python libraries:

```bash
pip install numpy pandas scikit-learn
```

### ▶️ Run the Notebook

1. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```
2. Open `Diabetes predic.ipynb`
3. Run all cells to execute the pipeline.

---

## 📈 Example Output

```python
Score: 0.7866 (Training Accuracy)
Score: 0.7727 (Testing Accuracy)

Prediction for input data:
[[ 0.3429808   1.41167241  0.14964075 -0.09637905  0.82661621 -0.78595734
   0.34768723  1.51108316]]
Prediction: [1]
Person is Diabetic
```

---

## ✍️ Author

**Hilary Giyane**
Data Analyst | Lusaka, Zambia
*Feel free to connect or share feedback!*

