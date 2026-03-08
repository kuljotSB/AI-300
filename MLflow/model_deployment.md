### Deploying Mlflow Model - Instructor Notes

create a custom environment with the following parameters and configs:
name - `iris-env`

parent image: `mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:latest`

conda.yaml file specs:
```yaml
dependencies:
  - python=3.10
  - pip:
      - mlflow-skinny==2.16.2
      - azureml-mlflow==1.57.0.post1
      - scikit-learn==1.5.2
      - azureml-inference-server-http==1.4.0
      - joblib==1.4.2
      - azure-ai-ml==1.20.0
      - azureml-defaults==1.59.0
      - azureml-ai-monitoring==1.0.0
      - pandas==2.2.3
name: iris-env
```

Use the following CURL Request:
```CURL
curl -X POST "YOUR_MODEL_SCORING_ENDPOINT" -H "Content-Type: application/json" -H "Authorization: Bearer YOUR_KEY" -d "{\"input_data\":[{\"Pregnancies\":8,\"Glucose\":85,\"BloodPressure\":65,\"SkinThickness\":29,\"Insulin\":0,\"BMI\":26.6,\"DiabetesPedigreeFunction\":0.672,\"Age\":34}]}"
```