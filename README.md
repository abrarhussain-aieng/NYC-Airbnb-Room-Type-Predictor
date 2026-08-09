# 🏠 NYC Airbnb Room Type Predictor

An end-to-end Machine Learning classification project that predicts the **Airbnb room type** of a New York City listing from listing characteristics such as location, price, minimum nights, review activity, host listing count, availability, and neighbourhood information.

The project covers the complete ML lifecycle:

**Data Analysis → Cleaning → Feature Engineering → Preprocessing → Model Comparison → Hyperparameter Tuning → Evaluation → Model Serialization → FastAPI API → Deployment**

## 🚀 Live Demo

**Live Application:**  
https://nyc-airbnb-room-type-predictor-imi8.onrender.com

The deployed application exposes a FastAPI-based prediction service. The API accepts listing features and returns the predicted room type together with class probabilities.

---

## 🎯 Project Objective

The objective is to build a multiclass classification model capable of predicting the Airbnb `room_type` from structured listing information.

The target variable contains three room categories:

- `Entire home/apt`
- `Private room`
- `Shared room`

The final trained model is packaged as a reusable scikit-learn pipeline and saved as `Model_Pipeline.pkl`.

---

## 📊 Dataset

The project uses the **AB_NYC_2019** Airbnb dataset.

### Dataset size

- **48,895 listings**
- **16 original columns**
- Target: `room_type`

### Original columns

| Column | Description |
|---|---|
| `id` | Listing identifier |
| `name` | Listing name |
| `host_id` | Host identifier |
| `host_name` | Host name |
| `neighbourhood_group` | NYC borough/group |
| `neighbourhood` | Specific neighbourhood |
| `latitude` | Listing latitude |
| `longitude` | Listing longitude |
| `room_type` | Target room category |
| `price` | Price per night |
| `minimum_nights` | Minimum nights required |
| `number_of_reviews` | Total reviews |
| `last_review` | Date of latest review |
| `reviews_per_month` | Average reviews per month |
| `calculated_host_listings_count` | Number of listings managed by host |
| `availability_365` | Availability during the year |

The notebook shows the dataset contains 48,895 rows and 16 columns. It also identifies missing values in `name`, `host_name`, `last_review`, and `reviews_per_month`. 

---

## 🧹 Data Cleaning & Preprocessing

The notebook follows a structured preprocessing workflow.

### 1. Removed non-predictive columns

The following columns were removed:

```text
id
name
host_id
host_name
last_review
```

### 2. Missing-value handling

`reviews_per_month` missing values were replaced with `0`.

The final ML pipeline also includes:

- Mean imputation for numerical features
- Most-frequent imputation for categorical features

### 3. Outlier treatment

The project caps:

- `price`
- `minimum_nights`

at their respective **99th percentiles** to reduce the influence of extreme values.

### 4. Feature separation

The final features are divided into numerical and categorical groups.

**Numerical features:**

```text
longitude
latitude
price
minimum_nights
number_of_reviews
reviews_per_month
calculated_host_listings_count
availability_365
```

**Categorical features:**

```text
neighbourhood_group
neighbourhood
```

### 5. Numerical preprocessing

```text
SimpleImputer(strategy="mean")
        ↓
StandardScaler()
```

### 6. Categorical preprocessing

```text
SimpleImputer(strategy="most_frequent")
        ↓
OneHotEncoder(handle_unknown="ignore")
```

A `ColumnTransformer` combines both preprocessing pipelines.

This keeps preprocessing and model inference consistent between training and production.

---

## 🔎 Exploratory Data Analysis

The notebook performs EDA using **Pandas, Matplotlib, and Seaborn**.

Analysis includes:

- Dataset structure and data types
- Descriptive statistics
- Missing-value analysis
- Duplicate-value checking
- Room-type distribution
- Numerical distributions
- Room type vs. price analysis
- Correlation analysis
- Geographic distribution of listings
- Latitude/longitude visualization by room type

The notebook reports **0 duplicate rows** in the dataset.

---

## 🤖 Model Development

Four classification algorithms were compared using 3-fold cross-validation.

| Model | CV Accuracy | CV Macro-F1 |
|---|---:|---:|
| Logistic Regression | 0.659 | 0.522 |
| Decision Tree | 0.782 | 0.650 |
| Random Forest | **0.850** | **0.717** |
| Gradient Boosting | **0.850** | 0.705 |

Random Forest was selected for further optimization because it achieved the strongest Macro-F1 performance among the evaluated models.

---

## ⚙️ Hyperparameter Optimization

`RandomizedSearchCV` was used to optimize the Random Forest pipeline.

### Search space

```python
{
    "classifier__n_estimators": [100, 200, 150, 300],
    "classifier__max_depth": [8, 12, 15, 20, None],
    "classifier__min_samples_split": [2, 5, 10]
}
```

Configuration:

- Cross-validation: `3-fold`
- Iterations: `10`
- Scoring: `f1_macro`
- Random state: `42`

### Best parameters

```text
n_estimators = 200
max_depth = None
min_samples_split = 10
```

Best cross-validation Macro-F1:

```text
0.7322
```

---

## 📈 Final Model Performance

The optimized Random Forest pipeline was evaluated on the held-out test set.

| Metric | Score |
|---|---:|
| Test Accuracy | **85.54%** |
| Test Macro-F1 | **73.58%** |

The notebook also generates a confusion matrix to inspect class-level prediction performance.

> **Note:** The reported metrics are based on the training notebook's recorded outputs and should be reproduced if the dataset, preprocessing, library versions, and random seeds are changed.

---

## 🧠 Final ML Pipeline

The production model is saved as:

```text
Model_Pipeline.pkl
```

The serialized pipeline contains the preprocessing and classifier workflow required for inference.

Conceptually:

```text
Raw Input
   │
   ├── Numerical Features
   │      ├── Mean Imputation
   │      └── Standard Scaling
   │
   ├── Categorical Features
   │      ├── Most-Frequent Imputation
   │      └── One-Hot Encoding
   │
   └──────────────┬──────────────
                  ↓
          Random Forest Classifier
                  ↓
        Room Type + Probabilities
```

---

## 🌐 FastAPI Backend

The production API is implemented with **FastAPI**.

The API:

- Loads `Model_Pipeline.pkl`
- Validates incoming request data with Pydantic
- Builds a Pandas DataFrame from the request
- Runs `model.predict()`
- Runs `model.predict_proba()`
- Returns the predicted room type and probabilities

The API source defines the `/predict` endpoint and the same ten production input features used by the trained pipeline.

### API endpoints

#### `GET /`

Health/status endpoint.

Example response:

```json
{
  "message": "Welcome to the NYC Airbnb Room Type Prediction API",
  "status": "API is running"
}
```

#### `POST /predict`

Prediction endpoint.

The API expects:

```text
latitude
longitude
price
minimum_nights
number_of_reviews
reviews_per_month
calculated_host_listings_count
availability_365
neighbourhood_group
neighbourhood
```

Example request:

```json
{
  "latitude": 40.75362,
  "longitude": -73.98377,
  "price": 225,
  "minimum_nights": 1,
  "number_of_reviews": 45,
  "reviews_per_month": 0.38,
  "calculated_host_listings_count": 2,
  "availability_365": 355,
  "neighbourhood_group": "Manhattan",
  "neighbourhood": "Midtown"
}
```

Example response structure:

```json
{
  "Predicted_room_type": "Entire home/apt",
  "Probability": [
    0.XX,
    0.XX,
    0.XX
  ]
}
```

The exact probability values depend on the input and the trained model.

---

## 🔐 Input Validation

The API uses Pydantic validation rules for important fields.

Examples include:

- Latitude: `-90` to `90`
- Longitude: `-180` to `180`
- Price: non-negative
- Minimum nights: `1` to `365`
- Number of reviews: non-negative
- Reviews per month: non-negative
- Host listing count: non-negative
- Availability: `0` to `365`
- Neighbourhood fields: non-empty strings

This provides a validation layer before data reaches the ML pipeline.

---

## 🗂️ Project Structure

Recommended GitHub structure:

```text
NYC-Airbnb-Room-Type-Predictor/
│
├── 📓 nyc_airbnb_room_type_classification.ipynb
├── 🐍 main.py
├── 🤖 Model_Pipeline.pkl
├── 📄 requirements.txt
├── 📄 README.md
└── 📊 AB_NYC_2019.csv              # optional / local dataset
```

### File responsibilities

| File | Purpose |
|---|---|
| `nyc_airbnb_room_type_classification.ipynb` | Complete EDA, preprocessing, model comparison, tuning and evaluation |
| `main.py` | FastAPI production API |
| `Model_Pipeline.pkl` | Serialized preprocessing + Random Forest model |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |
| `AB_NYC_2019.csv` | Training dataset, if included |

---

## 🛠️ Tech Stack

### Programming

- Python

### Data Science

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- scikit-learn
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- RandomizedSearchCV
- Cross-validation
- Pipeline
- ColumnTransformer

### Model Persistence

- Joblib

### API / Backend

- FastAPI
- Pydantic
- Pandas

### Deployment

- Render

---

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd NYC-Airbnb-Room-Type-Predictor
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI server

```bash
uvicorn main:app --reload
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

FastAPI's interactive API documentation can then be accessed at:

```text
http://127.0.0.1:8000/docs
```

---

## 📦 Recommended `requirements.txt`

```text
fastapi
uvicorn
pandas
numpy
scikit-learn
joblib
pydantic
```

For production reproducibility, pin the exact versions used during model training and deployment.

---

## 🧪 Testing the API

After starting the server, open:

```text
http://127.0.0.1:8000/docs
```

Use the interactive Swagger UI to:

1. Open `POST /predict`
2. Select **Try it out**
3. Enter listing information
4. Execute the request
5. Inspect the predicted room type
6. Inspect the returned probability distribution

---

## 🚀 Deployment

The application is deployed as a FastAPI service.

Deployment flow:

```text
GitHub Repository
       │
       ↓
Render
       │
       ├── FastAPI application
       ├── Model_Pipeline.pkl
       └── Python dependencies
       │
       ↓
Public Prediction API
```

### Production considerations

For a production deployment:

- Keep `Model_Pipeline.pkl` available to the application
- Install dependencies from `requirements.txt`
- Use a production ASGI server
- Configure environment-specific settings
- Avoid unrestricted CORS in sensitive production environments
- Pin ML library versions to maintain model compatibility

---

## 🔄 End-to-End ML Workflow

```text
AB_NYC_2019 Dataset
        │
        ↓
Data Loading
        │
        ↓
Data Inspection
        │
        ↓
EDA
        │
        ↓
Missing Value Analysis
        │
        ↓
Duplicate Check
        │
        ↓
Feature Cleaning
        │
        ├── Remove identifier/text columns
        ├── Fill missing review values
        └── Cap extreme price/night values
        │
        ↓
Train/Test Split
        │
        ↓
Preprocessing Pipeline
        │
        ├── Numerical → Impute → Scale
        └── Categorical → Impute → One-Hot Encode
        │
        ↓
Model Comparison
        │
        ├── Logistic Regression
        ├── Decision Tree
        ├── Random Forest
        └── Gradient Boosting
        │
        ↓
Random Forest Selected
        │
        ↓
RandomizedSearchCV
        │
        ↓
Final Evaluation
        │
        ↓
Model_Pipeline.pkl
        │
        ↓
FastAPI
        │
        ↓
POST /predict
        │
        ↓
Room Type + Probabilities
```

---

## ⭐ Key Highlights

- End-to-end supervised Machine Learning project
- Multiclass classification
- Real-world Airbnb dataset
- Exploratory Data Analysis
- Missing-value handling
- Outlier treatment
- Numerical feature scaling
- Categorical feature encoding
- `ColumnTransformer` preprocessing
- Scikit-learn `Pipeline`
- Multiple model comparison
- Cross-validation
- Randomized hyperparameter search
- Macro-F1 optimization
- Confusion matrix evaluation
- Joblib model serialization
- FastAPI REST API
- Pydantic request validation
- Cloud deployment
- Live prediction endpoint

---

## ⚠️ Limitations

This model is intended as a machine-learning demonstration and should not be interpreted as an official Airbnb classification system.

Potential limitations include:

- The dataset represents Airbnb listings from the 2019 NYC dataset.
- Market conditions and listing behaviour can change over time.
- The model uses structured listing attributes and does not use listing descriptions or images.
- Geographic and neighbourhood patterns may change.
- Model performance may vary on data that differs significantly from the training distribution.
- The reported performance should not be interpreted as a guarantee of real-world prediction accuracy.

---

## 🔮 Future Improvements

Possible extensions include:

- Add class-wise precision, recall and F1 reporting
- Perform deeper error analysis
- Add probability calibration
- Compare additional ensemble models
- Experiment with XGBoost or LightGBM
- Add feature importance and explainability with SHAP
- Add automated tests for the API
- Add Docker support
- Add CI/CD with GitHub Actions
- Add API authentication/rate limiting for production
- Add monitoring and logging
- Add data drift detection
- Create a dedicated frontend for interactive predictions
- Add automated model retraining

---

## 👨‍💻 Project Skills Demonstrated

This project demonstrates practical skills in:

```text
Python
Pandas
NumPy
Data Cleaning
EDA
Data Visualization
Feature Engineering
Scikit-learn
Classification
Model Selection
Cross-Validation
Hyperparameter Tuning
ML Pipelines
Model Serialization
FastAPI
REST APIs
Pydantic
Deployment
Production ML
```

---

## 📚 Project Files

The main project notebook contains the complete model-development workflow, while `main.py` provides the production FastAPI layer. The API loads the serialized `Model_Pipeline.pkl` and exposes `/predict` for inference.

---

## 📄 License

Add a license appropriate for your repository before publishing if you intend others to reuse, modify, or distribute the project.

---

## ⭐ Support

If this project helped you learn Machine Learning deployment, consider giving the repository a ⭐ on GitHub.

**Built as an end-to-end Machine Learning project — from raw Airbnb data to a deployed prediction API.**
