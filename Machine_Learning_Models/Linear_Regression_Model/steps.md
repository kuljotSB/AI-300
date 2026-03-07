### For Instructor Use Only

![final_pipeline](./images/final_pipeline.png)

https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-designer-automobile-price-train-score?view=azureml-api-1

### Creating Custom Environment

Parent Image: mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04

Conda File Definition:
```yaml
name: project_environment
channels:
  - conda-forge
dependencies:
  - python=3.8.10
  - pip=20.2
  - pip:
      - azureml-inference-server-http==1.3.4
      - azureml-designer-classic-modules==0.0.182
      - azureml-designer-serving==0.0.13
```
