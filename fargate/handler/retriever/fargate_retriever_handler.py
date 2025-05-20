import os
import json
from fargate.retriever_processor import RetrieverProcessor
from flotorch_core.logger.global_logger import get_logger
from flotorch_core.config.config import Config
from flotorch_core.config.env_config_provider import EnvConfigProvider

logger = get_logger()

# Initialize configuration provider and config
env_config_provider = EnvConfigProvider()
config = Config(env_config_provider)


def get_environment_data():
    """
    Fetches task token and input data from environment variables.
    Returns:
        tuple: Task token (str) and input data (dict).
    """
    task_token = config.get_task_token()

    input_data = config.get_fargate_input_data()
    try:
        input_data = json.loads(input_data) if isinstance(input_data, str) else input_data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in INPUT_DATA: {str(e)}")

    return task_token, input_data



def main():
    """
    Main entry point for the Fargate retriever handler.
    """
    try:
        # task_token, input_data = get_environment_data()

        EXPERIMENT_ID = os.environ.get("EXPERIMENT_ID")
        print(f"EXPERIMENT_ID: {EXPERIMENT_ID}")
        EXECUTION_ID = os.environ.get("EXECUTION_ID")
        print(f"EXECUTION_ID: {EXECUTION_ID}")

        # Initialize and process the RetrieverProcessor
        fargate_processor = RetrieverProcessor(execution_id=EXECUTION_ID, experiment_id=EXPERIMENT_ID)
        fargate_processor.process()
    except Exception as e:
        logger.error(f"Error processing event: {str(e)}")
        raise


if __name__ == "__main__":
    main()
