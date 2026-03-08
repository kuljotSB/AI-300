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


create the scoring_script as following:
```python
"""
Scoring script for an Azure ML managed online endpoint with monitoring.
"""

from azureml.ai.monitoring import Collector
from azureml.ai.monitoring.context import BasicCorrelationContext
from azureml.contrib.services.aml_request import rawhttp
from azureml.contrib.services.aml_response import AMLResponse
import logging
import os
import pandas as pd
import json
import uuid
import mlflow.pyfunc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.propagate = True


def init():
    """
    Called once when the container starts.
    Loads the MLflow model and initializes monitoring collectors.
    """
    global model, inputs_collector, outputs_collector

    # Initialize collectors
    inputs_collector = Collector(
        name="model_inputs",
        on_error=lambda e: logging.info(f"collector error: {e}")
    )

    outputs_collector = Collector(
        name="model_outputs",
        on_error=lambda e: logging.info(f"collector error: {e}")
    )

    # Load MLflow model
    model = _load_model()

    logger.info("Init completed successfully")


def _load_model():
    """
    Load MLflow model from Azure ML mounted directory.
    """
    model_dir = os.getenv("AZUREML_MODEL_DIR")

    # Because you logged with artifact_path="model"
    model_path = os.path.join(model_dir, "model")

    model = mlflow.pyfunc.load_model(model_path)

    logger.info(f"Model loaded from {model_path}")
    return model


@rawhttp
def run(raw_data):
    """
    Called for every scoring request.
    """

    try:
        logger.info("Request received")

        # Parse request body
        raw_data = raw_data.get_data().decode("utf-8")
        payload = json.loads(raw_data)

        data = payload["input_data"]

        # Convert to DataFrame
        input_df = pd.DataFrame(data)

        logger.info(f"Input dataframe:\n{input_df}")

        # Correlation context for monitoring
        artificial_context = BasicCorrelationContext(id=str(uuid.uuid4()))

        # Collect inputs
        context = inputs_collector.collect(input_df, artificial_context)

        # Run prediction
        predictions = model.predict(input_df)

        # Collect outputs
        outputs_collector.collect(predictions, context)

        logger.info(f"Prediction response: {predictions}")

        # Return JSON response
        response = {"predictions": predictions.tolist()}

        return AMLResponse(json.dumps(response), status_code=200)

    except Exception as error:
        logger.error(f"Error during prediction: {repr(error)}")

        return AMLResponse(
            json.dumps({"error": repr(error)}),
            status_code=400
        )
```

Use the following CURL Request:
```CURL
curl -X POST "YOUR_MODEL_SCORING_ENDPOINT" -H "Content-Type: application/json" -H "Authorization: Bearer YOUR_KEY" -d "{\"input_data\":[{\"Pregnancies\":8,\"Glucose\":85,\"BloodPressure\":65,\"SkinThickness\":29,\"Insulin\":0,\"BMI\":26.6,\"DiabetesPedigreeFunction\":0.672,\"Age\":34}]}"
```