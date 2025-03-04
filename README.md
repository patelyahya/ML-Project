# ML-Project

## Project Title and Description

This project is an end-to-end machine learning project implemented with MLOps using MLFlow for the complete lifecycle. The project includes data preprocessing, model training, and evaluation, all integrated with MLFlow for tracking experiments, parameters, and metrics.

## Setup Instructions

### Dependencies

- Python 3.x
- MLFlow
- scikit-learn
- pandas
- numpy

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/patelyahya/ML-Project.git
   cd ML-Project
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Instructions

### Running the Project

1. Preprocess the data:
   ```bash
   python src/data_preprocessing/DataPreprocessor.py
   ```

2. Train the model:
   ```bash
   python src/model_training/ModelTrainer.py
   ```

3. Evaluate the model:
   ```bash
   python src/model_evaluation/ModelEvaluator.py
   ```

### Using MLFlow

MLFlow is used to track experiments, parameters, and metrics throughout the project. To start the MLFlow UI, run:
```bash
mlflow ui
```
This will start the MLFlow UI at `http://localhost:5000`, where you can view and compare your experiments.

## MLFlow Integration and Benefits

MLFlow is integrated into the project to provide the following benefits:

- **Experiment Tracking**: Track and compare different experiments, including parameters, metrics, and artifacts.
- **Model Management**: Save and load models using MLFlow, making it easy to manage different versions of your models.
- **Reproducibility**: Ensure that experiments are reproducible by logging all relevant information, including code, data, and environment.
