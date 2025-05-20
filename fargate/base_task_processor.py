import json
from abc import ABC, abstractmethod
import sys
from botocore.exceptions import ClientError
from flotorch_core.logger.global_logger import get_logger
import boto3

logger = get_logger()

class BaseFargateTaskProcessor(ABC):
    """
    Abstract base class for Fargate task processors.
    """

    def __init__(self, execution_id, experiment_id):
        """
        Initializes the task processor with task token and input data.
        Args:
            task_token (str): The Step Functions task token.
            input_data (dict): The input data for the task.
        """
        # self.task_token = task_token
        # self.input_data = input_data
        self.execution_id = execution_id
        self.experiment_id = experiment_id

    @abstractmethod
    def process(self):
        """
        Abstract method to be implemented by subclasses for processing tasks.
        """
        raise NotImplementedError("Subclasses must implement the process method.")

    def send_task_success(self, output: dict):
        """
        Sends task success signal to Step Functions.
        Args:
            output (dict): The output data to send to Step Functions.
        """
        print(json.dumps({"status": "success", "output": output}))
        sys.exit(0)

    def send_task_failure(self, error_message: str):
        """
        Sends task failure signal to Step Functions.
        Args:
            error_message (str): The error message to send to Step Functions.
        """
        print(json.dumps({"status": "error", "output": error_message}))
        sys.exit(1) 