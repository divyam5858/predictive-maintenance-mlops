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
