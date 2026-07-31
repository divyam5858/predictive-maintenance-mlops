# Predictive Maintenance MLOps

This repository contains the practical implementation developed alongside the textbook **Practical MLOps with ZenML and MLflow**. Throughout the book, a Predictive Maintenance System is developed incrementally while introducing production-ready MLOps concepts, tools, and best practices.

---

## Technology Stack

- Python 3.11
- Git
- Docker
- ZenML
- MLflow
- AWS CLI
- Jupyter Notebook
- Visual Studio Code

---

## Project Structure

```
predictive-maintenance-mlops/
├── artifacts/
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── pipelines/
├── src/
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Environment Setup

Clone the repository:

```bash
git clone https://github.com/divyam5858/predictive-maintenance-mlops.git
cd predictive-maintenance-mlops
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

**Windows (Git Bash)**

```bash
source .venv/Scripts/activate
```

**Windows (Command Prompt)**

```cmd
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

---

## Current Progress

### ✅ Chapter 2 — Production ML Environment Setup

Completed:

- Development environment setup
- Python virtual environment
- Git repository initialization
- Docker installation and verification
- ZenML installation and initialization
- MLflow installation and verification
- AWS CLI installation
- Jupyter Notebook configuration
- Visual Studio Code configuration
- Dependency management
- Project directory structure

### 🚀 Upcoming

**Chapter 3 — Building the First End-to-End Machine Learning Pipeline with ZenML**

The next phase introduces ZenML pipelines by implementing a complete end-to-end workflow for the Predictive Maintenance System.

---

## License

This repository accompanies the textbook **Practical MLOps with ZenML and MLflow** and is intended for educational purposes.