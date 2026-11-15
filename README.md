# TumorAI 🧬

**An end-to-end breast cancer prediction application powered by Machine Learning and FastAPI.**

TumorAI is a web application that uses a trained Machine Learning model to predict whether a breast tumor is likely to be **Benign** or **Malignant** based on 30 diagnostic features.

The project demonstrates an end-to-end ML workflow, from model development and preprocessing to API development and frontend integration.

> **Disclaimer:** TumorAI is an educational project. Its predictions are not medical diagnoses and must not be used for clinical decision-making.

## Features

- Breast tumor classification using Logistic Regression.
- Scikit-learn Pipeline with StandardScaler and LogisticRegression.
- 30 diagnostic input features.
- Prediction probabilities for the malignant class.
- REST API built with FastAPI.
- Dynamic frontend feature generation.
- Demo input functionality using a sample from the dataset.
- Input validation and error handling.
- Automated API tests using Pytest.

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model Serialization | Joblib |
| Backend | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| API Server | Uvicorn |
| Testing | Pytest, HTTPX |

## Project Structure

```text
TumorAI/
├── backend/
│   ├── __init__.py
│   ├── main.py
│   └── breast_cancer_pipeline.pkl
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── notebooks/
│   └── model_training.ipynb
├── tests/
│   └── test_api.py
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Machine Learning Workflow

1. Load and inspect the breast cancer dataset.
2. Separate input features and target labels.
3. Remove irrelevant columns such as `id`.
4. Encode diagnosis labels:
   - `B` → 0 (Benign)
   - `M` → 1 (Malignant)
5. Split the dataset into training and testing sets.
6. Apply feature standardization using StandardScaler.
7. Train a Logistic Regression classifier.
8. Save the complete preprocessing and prediction pipeline using Joblib.

The trained pipeline is used by the FastAPI backend for predictions.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd TumorAI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Linux/macOS**
```bash
source .venv/bin/activate
```

**Windows**
```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For development and testing:

```bash
pip install -r requirements-dev.txt
```

## Dataset

The original dataset was obtained from Kaggle.

The dataset file is excluded from this repository because its redistribution license has not yet been verified.

To use the demo endpoint:

1. Obtain the dataset from its original source.
2. Place the CSV in the project root.
3. Name it `breast-cancer.csv`.

The dataset must contain all 30 model features used during training.

## Running the Application

### 1. Start the FastAPI backend

From the project root:

```bash
uvicorn backend.main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### 2. Start the frontend

Open another terminal from the project root:

```bash
python -m http.server 5500 --directory frontend
```

Open:

```text
http://127.0.0.1:5500
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Check API and model status |
| GET | `/features` | Retrieve required model features |
| POST | `/predict` | Generate a prediction |
| GET | `/demo` | Get a sample input from the dataset |

## Testing

Run the automated tests from the project root:

```bash
python -m pytest -v
```

The tests cover:

- API availability.
- Health status.
- Feature retrieval.
- Valid predictions.
- Missing input validation.
- Unexpected input validation.

## Model Limitations

- The model is trained on a specific dataset and may not generalize to other populations or clinical settings.
- Predictions depend on the quality and distribution of input data.
- Model-estimated probabilities are not equivalent to clinically validated cancer risk.
- This project is intended for educational and demonstration purposes only.

## Future Improvements

- Improve model evaluation and validation.
- Add model comparison and hyperparameter tuning.
- Introduce persistent prediction history.
- Improve deployment and production configuration.
- Add monitoring for model performance and data drift.

## Author

**Ramesh Dhamala**

Engineering Student | Machine Learning & Deep Learning Enthusiast
