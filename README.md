
````markdown
# Predictive Maintenance MLOps

This repository contains the practical implementation developed alongside the textbook **Practical MLOps with ZenML and MLflow**.

Throughout the book, a **Predictive Maintenance System** is developed incrementally while introducing production-ready MLOps concepts, tools, and best practices. Each chapter extends the existing system with additional machine learning and MLOps capabilities.

---

## Technology Stack

- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- Git
- Docker
- ZenML
- MLflow
- Great Expectations
- AWS CLI
- Jupyter Notebook
- Visual Studio Code

---

## Project Structure

```text
predictive-maintenance-mlops/
├── artifacts/
├── configs/
│   └── pipeline_config.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── pipelines/
│   ├── predictive_maintenance_pipeline.py
│   └── run_pipeline.py
├── src/
│   ├── data_ingestion.py
│   ├── data_cleaning.py
│   ├── data_validation.py
│   ├── gx_validation.py
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   ├── feature_preprocessing.py
│   ├── feature_pipeline.py
│   ├── feature_consistency.py
│   ├── feature_artifact.py
│   ├── feature_store.py
│   ├── ingestion_step.py
│   ├── cleaning_step.py
│   ├── validation_step.py
│   ├── feature_engineering_step.py
│   ├── feature_preprocessing_step.py
│   ├── training_step.py
│   └── inference_step.py
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
````

---

## Chapter Progress

### Chapter 2 — Production ML Environment Setup

Established the foundational development environment for the Predictive Maintenance System.

Implemented:

* Python virtual environment
* Git repository initialization
* Project directory structure
* Dependency management
* Docker installation and verification
* ZenML installation and configuration
* MLflow installation and verification
* AWS CLI setup
* Jupyter Notebook configuration
* Visual Studio Code configuration

---

### Chapter 3 — Building the First End-to-End Machine Learning Pipeline with ZenML

Implemented the first end-to-end machine learning workflow using ZenML.

Implemented:

* Data ingestion
* Data cleaning
* Feature preparation
* Model training
* Model inference
* ZenML pipeline orchestration
* Initial MLflow integration
* End-to-end pipeline execution

This chapter established the basic machine learning workflow that was extended in subsequent chapters.

---

### Chapter 4 — Reproducible ZenML Pipeline Architecture

Extended the initial pipeline into a structured and reproducible ZenML architecture.

Implemented:

* Modular ZenML pipeline steps
* Centralized pipeline configuration
* Reproducible train/test splitting
* Random state management
* Structured training and inference steps
* Reusable machine learning components
* Pipeline-level configuration
* Pipeline execution and validation

The resulting architecture provides the foundation for data validation, feature engineering, experiment tracking, and model management.

---

### Chapter 5 — Data Ingestion and Validation with Pandas, NumPy, and Great Expectations

Implemented a structured data engineering and validation layer for the Predictive Maintenance System.

Implemented:

* Data ingestion using Pandas
* Data exploration and profiling
* Data cleaning and preprocessing
* Missing-value validation
* Duplicate-record validation
* Data-type validation
* Numeric-value validation
* Machine identifier validation
* Sensor-value validation
* IQR-based outlier detection
* Schema validation
* Data contracts
* Great Expectations validation
* Automated data-quality checks
* ZenML data-validation integration

The predictive maintenance dataset contains industrial sensor and operational information including:

* Timestamp
* Machine ID
* Ambient temperature
* Humidity
* Temperature
* Vibration
* Pressure
* Rotational speed
* Torque
* Operating hours
* Failure status

The validation layer ensures that reliable and structurally consistent data is passed to subsequent stages of the machine learning workflow.

---

### Chapter 6 — Feature Engineering and Data Preparation for Predictive Maintenance Models

Implemented a reusable feature engineering and preprocessing layer using Pandas, NumPy, and Scikit-learn.

Implemented:

* Domain-driven feature engineering
* Time-based feature creation
* Sensor-based feature creation
* Feature selection
* Feature transformation
* Numerical preprocessing
* Categorical encoding
* Feature scaling
* Reusable feature pipelines
* Feature consistency checks
* Feature artifacts
* Feature Store concepts
* Training and production feature consistency
* ZenML feature engineering integration

The feature pipeline transforms validated industrial sensor data into a structured, model-ready representation for predictive maintenance.

---

### Chapter 7 — Experiment Tracking, Model Versioning, and Metadata Management with MLflow

Implemented MLflow as the experiment tracking and model-management layer of the Predictive Maintenance System.

Implemented:

* MLflow Tracking Server configuration
* MLflow experiment organization
* Parameter logging
* Metric logging
* Model signatures
* Artifact logging
* Classification reports
* Confusion matrices
* Feature information tracking
* Trained model logging
* Experimental run comparison
* Experiment history management
* Model versioning
* MLflow Model Registry
* Registered-model metadata
* Model-version tags
* Model aliases
* Model lifecycle roles
* MLflow and ZenML integration
* Reproducibility verification

The MLflow experiment used by the project is:

```text
Predictive Maintenance
```

The registered model is:

```text
predictive-maintenance-model
```

Multiple Random Forest configurations were evaluated using different numbers of estimators. Their parameters, metrics, artifacts, model signatures, and trained models were recorded through MLflow.

The MLflow and ZenML integration allows experiment information to be recorded directly as part of the automated machine learning pipeline.

---

## Current Machine Learning Workflow

The system currently follows this workflow:

```text
Raw Industrial Data
        ↓
Data Ingestion
        ↓
Data Cleaning
        ↓
Data Validation
        ↓
Great Expectations Validation
        ↓
Feature Engineering
        ↓
Feature Selection
        ↓
Feature Transformation
        ↓
Train/Test Split
        ↓
Model Training
        ↓
MLflow Experiment Tracking
        ↓
Model Evaluation
        ↓
Model Registration
        ↓
Model Versioning
        ↓
Prediction
```

ZenML is used to orchestrate the machine learning workflow, while MLflow provides experiment tracking, model logging, and model lifecycle management.

---

## Testing

The project includes an automated test suite covering the implemented data, feature, pipeline, and MLflow functionality.

The test suite is executed as new chapters extend the Predictive Maintenance System to help maintain the reliability of previously implemented components.

---

## Upcoming

### Chapter 8 — Model Training, Evaluation, and Benchmarking with Scikit-Learn, XGBoost, and LightGBM

The next stage of the project focuses on systematic model development, evaluation, and benchmarking.

Planned implementation includes:

* Model development lifecycle
* Scikit-learn model training
* XGBoost model training
* LightGBM model training
* Predictive maintenance evaluation metrics
* Cross-validation strategies
* Hyperparameter tuning
* Model benchmarking
* SHAP-based model explainability
* Feature importance analysis
* Model comparison
* Model acceptance criteria
* MLflow evaluation tracking
* ZenML integration

Chapter 8 will extend the existing feature engineering and experiment-tracking infrastructure by training and comparing multiple candidate machine learning models before deployment.

---

## Repository and Textbook

This repository accompanies the textbook:

**Practical MLOps with ZenML and MLflow**

The implementation and textbook are developed together so that the concepts introduced in each chapter are demonstrated through the continuous **Predictive Maintenance System** case study.

```

