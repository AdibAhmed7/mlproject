# 🎓 Student Exam Performance Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-enabled-189fdd)
![CatBoost](https://img.shields.io/badge/CatBoost-enabled-ffcc00)

An end-to-end machine learning web application that predicts a student's **mathematics score** based on demographic and academic background features. The project follows a production-style modular architecture — from raw data ingestion through preprocessing, hyperparameter-tuned model training, and a Flask-powered prediction UI.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [ML Pipeline](#-ml-pipeline)
- [Models & Evaluation](#-models--evaluation)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Known Issues](#-known-issues)
- [License](#-license)

---

## 🔍 Overview

This project predicts a student's **math score** (0–100) using the following input features:

| Feature | Type | Example |
|---|---|---|
| Gender | Categorical | `male`, `female` |
| Race / Ethnicity | Categorical | `group A` – `group E` |
| Parental Level of Education | Categorical | `bachelor's degree`, `master's degree`, etc. |
| Lunch Type | Categorical | `standard`, `free/reduced` |
| Test Preparation Course | Categorical | `completed`, `none` |
| Reading Score | Numerical | 0 – 100 |
| Writing Score | Numerical | 0 – 100 |

---

## 🖥️ Demo

Once running locally, navigate to `http://localhost:5000/predictdata` to access the prediction form:

1. Fill in the student's demographic and academic details
2. Enter their reading and writing scores
3. Click **Predict Mathematics Score** to get the model's prediction

---

## 📁 Project Structure

```
mlproject/
│
├── artifacts/                        # Auto-generated during training
│   ├── data.csv                      # Full raw dataset copy
│   ├── train.csv                     # 80% training split
│   ├── test.csv                      # 20% test split
│   ├── preprocessor.pkl              # Fitted ColumnTransformer
│   └── model.pkl                     # Best trained model
│
├── notebook/
│   ├── data/
│   │   └── stud.csv                  # Source dataset
│   ├── 1 . EDA STUDENT PERFORMANCE .ipynb
│   └── 2. MODEL TRAINING.ipynb
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py         # Load CSV → train/test split → save to artifacts/
│   │   ├── data_transformation.py    # Build & fit preprocessing pipeline
│   │   └── model_trainer.py          # Train 7 models with GridSearchCV, save best
│   │
│   ├── pipeline/
│   │   ├── predict_pipeline.py       # Load artifacts, transform input, return prediction
│   │   └── train_pipeline.py         # Orchestrates the full training flow
│   │
│   ├── exception.py                  # Custom exception with file/line info
│   ├── logger.py                     # Timestamped file logging
│   └── utils.py                      # save_object, load_object, evaluate_models
│
├── templates/
│   ├── index.html                    # Landing page
│   └── home.html                     # Prediction form & result display
│
├── app.py                            # Flask application entry point
├── setup.py                          # Package installer (finds src/ as a package)
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## 📊 Dataset

**Source:** `notebook/data/stud.csv`  
**Records:** ~1,000 students  
**Target column:** `math_score` (continuous, 0–100)

The dataset includes student performance scores in three subjects alongside demographic information. It is a commonly used educational dataset for regression tasks.

---

## ⚙️ ML Pipeline

### 1. Data Ingestion (`data_ingestion.py`)
- Reads `stud.csv` from `notebook/data/`
- Performs an 80/20 train-test split (`random_state=42`)
- Saves `data.csv`, `train.csv`, and `test.csv` to `artifacts/`

### 2. Data Transformation (`data_transformation.py`)
A `ColumnTransformer` with two sub-pipelines:

**Numerical features** (`reading_score`, `writing_score`):
```
SimpleImputer(strategy="median") → StandardScaler()
```

**Categorical features** (`gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`):
```
SimpleImputer(strategy="most_frequent") → OneHotEncoder() → StandardScaler(with_mean=False)
```

The fitted preprocessor is serialised to `artifacts/preprocessor.pkl`.

### 3. Model Training (`model_trainer.py`)
Seven regression models are trained with `GridSearchCV` (3-fold CV). The best model by **R² score on the test set** is saved to `artifacts/model.pkl`.

---

## 🤖 Models & Evaluation

| Model | Hyperparameters Tuned |
|---|---|
| Linear Regression | — |
| Decision Tree Regressor | `criterion` |
| Random Forest Regressor | `n_estimators` |
| Gradient Boosting Regressor | `learning_rate`, `subsample`, `n_estimators` |
| XGBoost Regressor | `learning_rate`, `n_estimators` |
| CatBoost Regressor | `depth`, `learning_rate`, `iterations` |
| AdaBoost Regressor | `learning_rate`, `n_estimators` |

The model with the highest test R² score (minimum threshold: **0.6**) is automatically selected and saved. If no model meets the threshold, a `CustomException` is raised.

---

## 🛠 Tech Stack

| Category | Library / Tool |
|---|---|
| Language | Python 3.8+ |
| Web Framework | Flask |
| ML & Preprocessing | scikit-learn |
| Boosting | XGBoost, CatBoost, AdaBoost |
| Data Handling | pandas, numpy |
| Visualisation | matplotlib, seaborn (notebooks) |
| Serialisation | pickle, dill |
| Hyperparameter Tuning | GridSearchCV |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/AdibaAhmed07/mlproject.git
cd mlproject
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install as a package (which also installs all dependencies):

```bash
pip install -e .
```

### 4. Run the Training Pipeline

```bash
python src/components/data_ingestion.py
```

This will:
- Ingest and split the raw dataset
- Fit the preprocessing pipeline
- Train and evaluate all 7 models with GridSearchCV
- Save `preprocessor.pkl` and `model.pkl` to `artifacts/`

### 5. Start the Web Application

```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

---

## 🧪 Usage

**Home page:** `http://localhost:5000/`  
**Prediction form:** `http://localhost:5000/predictdata`

Fill in all student fields and submit. The predicted maths score will appear below the form.

**Notebook exploration:**  
Open the Jupyter notebooks in `notebook/` for full EDA and model training experiments:

```bash
jupyter notebook notebook/
```

---

## ⚠️ Known Issues

- **Windows path hardcoding:** `data_ingestion.py` uses `r'notebook\data\stud.csv'` (Windows backslash). On macOS/Linux, update this to `'notebook/data/stud.csv'` or use `os.path.join()`.
- **`predict_pipeline.py` paths:** Artifact paths use Windows-style backslashes (`r"artifacts\model.pkl"`). Update to forward slashes or `os.path.join()` for cross-platform compatibility.
- **`utils.py` duplicate function:** `load_object` is defined twice. The second definition (using `dill`) is unreachable due to incorrect indentation — it sits inside the first function's `except` block. The first definition (using `pickle`) is what actually runs.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is open source. See the repository for license details.

---

*Built with ❤️ by [Adiba Ahmed](https://github.com/AdibaAhmed07)*
