from datetime import datetime
import json
from typing import Dict, List, Optional, Union, Any
import uuid
from flotorch_core.embedding.embedding_registry import embedding_registry
from flotorch_core.embedding.guardrails.guardrails_embedding import GuardrailsEmbedding
from fargate.base_task_processor import BaseFargateTaskProcessor
from flotorch_core.guardrails.guardrails import BedrockGuardrail
from flotorch_core.inferencer.guardrails.guardrails_inferencer import GuardRailsInferencer
from flotorch_core.inferencer.inferencer_provider_factory import InferencerProviderFactory
from flotorch_core.logger.global_logger import get_logger
from flotorch_core.config.config import Config
from flotorch_core.config.env_config_provider import EnvConfigProvider
from flotorch_core.reader.json_reader import JSONReader
from flotorch_core.rerank.rerank import BedrockReranker
from retriever.retriever import Retriever
from flotorch_core.storage.db.dynamodb import DynamoDB
from flotorch_core.storage.db.postgresdb import PostgresDB
from flotorch_core.storage.db.vector.guardrails_vector_storage import GuardRailsVectorStorage
from flotorch_core.storage.db.vector.open_search import OpenSearchClient
from flotorch_core.inferencer.bedrock_inferencer import BedrockInferencer
from flotorch_core.storage.db.vector.vector_storage_factory import VectorStorageFactory
from flotorch_core.storage.db.vector.reranked_vector_storage import RerankedVectorStorage
from flotorch_core.storage.storage_provider_factory import StorageProviderFactory

from flotorch_core.embedding.titanv2_embedding import TitanV2Embedding
from flotorch_core.embedding.titanv1_embedding import TitanV1Embedding
from flotorch_core.embedding.cohere_embedding import CohereEmbedding
from flotorch_core.embedding.bge_large_embedding import BGELargeEmbedding, BGEM3Embedding, GTEQwen2Embedding


logger = get_logger()
env_config_provider = EnvConfigProvider()
config = Config(env_config_provider)

def read_json_data_by_url(storage_provider, path:str, headers: dict = None) -> dict:
    data = "".join(chunk.decode("utf-8") for chunk in storage_provider.read(path, headers))
    return json.loads(data)

class RetrieverProcessor(BaseFargateTaskProcessor):
    """
    Processor for retriever tasks in Fargate.
    """

    def process(self):
        logger.info("Starting retriever process.")
        try:
            execution_id = self.execution_id
            experiment_id = self.experiment_id
            if self.config_data:
                config_info = self.config_data.get("console", {})
                config_base_url = config_info.get("url", "").rstrip("/")
                config_routes = config_info.get("endpoints", {})
                config_headers = config_info.get("headers", {})
                get_config_url = config_base_url + config_routes.get("experiment", "")
                storage_provider = StorageProviderFactory.create_storage_provider(get_config_url)
                get_exp_config_data = read_json_data_by_url(storage_provider, get_config_url, config_headers)
                exp_config_data = get_exp_config_data.get("config", {})
            else:
                exp_config_data = get_experiment_config(execution_id, experiment_id)
                print(f"exp_config_data: {exp_config_data}")
            if not exp_config_data:
                raise ValueError(
                    f"Experiment configuration not found for execution_id: {execution_id} and experiment_id: {experiment_id}")

            gt_data = exp_config_data.get("gt_data")
            storage = StorageProviderFactory.create_storage_provider(gt_data)
            gt_data_path = storage.get_path(gt_data)
            json_reader = JSONReader(storage)

            if exp_config_data.get("knowledge_base", False) and not exp_config_data.get("bedrock_knowledge_base", False):
                embedding_class = embedding_registry.get_model(exp_config_data.get("embedding_model"))
                embedding = embedding_class(
                    exp_config_data.get("embedding_model"), 
                    exp_config_data.get("aws_region"), 
                    int(exp_config_data.get("vector_dimension")))
                is_opensearch_required = True
            else:
                embedding = None
                is_opensearch_required = False
            
            if exp_config_data.get("enable_guardrails", False):
                base_guardrails = BedrockGuardrail(exp_config_data.get("guardrail_id", ""), exp_config_data.get("guardrail_version", 0), exp_config_data.get("aws_region", "us-east-1"))
            
            
            vector_storage = VectorStorageFactory.create_vector_storage(
                knowledge_base=exp_config_data.get("knowledge_base", False),
                use_bedrock_kb=exp_config_data.get("bedrock_knowledge_base", False),
                embedding=embedding,
                opensearch_host=config.get_opensearch_host() if is_opensearch_required else None,
                opensearch_port=config.get_opensearch_port() if is_opensearch_required else None,
                opensearch_username=config.get_opensearch_username() if is_opensearch_required else None,
                opensearch_password=config.get_opensearch_password() if is_opensearch_required else None,
                index_id=exp_config_data.get("index_id"),
                knowledge_base_id=exp_config_data.get("kb_data"),
                aws_region=exp_config_data.get("aws_region")
            )

            if exp_config_data.get("enable_guardrails", False) and exp_config_data.get("enable_prompt_guardrails", False):
                vector_storage = GuardRailsVectorStorage(
                    vector_storage, 
                    base_guardrails,
                    exp_config_data.get("enable_prompt_guardrails", False),
                    exp_config_data.get("enable_context_guardrails", False)
                )

            reranker = BedrockReranker(exp_config_data.get("aws_region"), exp_config_data.get("rerank_model_id")) \
                if exp_config_data.get("rerank_model_id").lower() != "none" \
                else None
            if reranker:
                vector_storage = RerankedVectorStorage(vector_storage, reranker)

            inferencer = InferencerProviderFactory.create_inferencer_provider(
                exp_config_data.get("gateway_enabled", False),
                f'{exp_config_data.get("gateway_url", "")}/api/openai/v1',
                exp_config_data.get("gateway_api_key", ""),
                exp_config_data.get("retrieval_service"),
                exp_config_data.get("retrieval_model"), 
                exp_config_data.get("aws_region"), 
                config.get_sagemaker_arn_role(),
                int(exp_config_data.get("n_shot_prompts", 0)),
                float(exp_config_data.get("temp_retrieval_llm", 0)), 
                exp_config_data.get("n_shot_prompt_guide_obj")
            )
            
            if exp_config_data.get("enable_guardrails", False) and exp_config_data.get("enable_response_guardrails", False):
                inferencer = GuardRailsInferencer(inferencer, base_guardrails)

            retriever = Retriever(json_reader, embedding, vector_storage, inferencer, reranker)
            hierarchical = exp_config_data.get("chunking_strategy") == 'hierarchical'
            result, total_input_tokens, total_output_tokens = retriever.retrieve(
                gt_data_path, 
                "What is the patient's name?",
                int(exp_config_data.get("knn_num")), 
                hierarchical
            )

            # metrics
            batch_items = []
            for item in result:
                metrics = create_metrics(
                    exp_config_data.get("execution_id"),
                    exp_config_data.get("experiment_id"),
                    item.question,
                    item.gt_answer,
                    item.answer,
                    item.reference_contexts,
                    item.query_metadata,
                    item.answer_metadata,
                    item.guardrails_input_assessment,
                    item.guardrails_context_assessment,
                    item.guardrails_output_assessment,
                    exp_config_data.get("guardrail_id", ""),
                    item.guardrails_block_level,
                    item.guardrails_blocked
                )                

                batch_items.append(metrics)
                if len(batch_items) >= 25:
                    write_batch_to_metrics_db(batch_items, self.config_data, execution_id, experiment_id)
                    batch_items = []

            if len(batch_items) > 0:
                write_batch_to_metrics_db(batch_items, self.config_data, execution_id, experiment_id)

            output = {"status": "success", "message": "Retriever completed successfully."}
            self.send_task_success(output)
        except Exception as e:
            logger.error(f"Error during retriever process: {str(e)}")
            self.send_task_failure(str(e))


def get_experiment_config(execution_id, experiment_id) -> Dict:
    """
    Retrieves the experiment configuration from the dynamo db.
    """
    db_type = config.get_db_type()
    if db_type == "POSTGRESDB":
        db = PostgresDB(
            dbname=config.get_postgress_db(),
            user=config.get_postgress_user(),
            password=config.get_postgress_password(),
            host=config.get_postgress_host(),
            port=config.get_postgress_port()
        )
    else:
        db = DynamoDB(config.get_experiment_table_name())

    data = db.read({"id": experiment_id, "execution_id": execution_id})
    if data:
        experiment_config = data[0].get("config", {}) if isinstance(data, list) else data.get("config", {})
        return experiment_config
    return None

# metrics: to be removed later
def create_metrics(
    execution_id,
    experiment_id,
    question: str,
    gt_answer: str,
    answer: str,
    reference_contexts: List[str],
    query_metadata: Dict[str, int],
    answer_metadata: Dict[str, int],
    guardrail_input_assessment: Optional[Union[List[Dict], Dict]] = None,
    guardrail_context_assessment: Optional[Union[List[Dict], Dict]] = None,
    guardrail_output_assessment: Optional[Union[List[Dict], Dict]] = None,
    guardrail_id: Optional[str] = None,
    guardrails_block_level: Optional[str] = '',
    guardrails_blocked: Optional[bool] = False
):
    metrics = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "execution_id": execution_id,
        "experiment_id": experiment_id,
        "question": question,
        "gt_answer": gt_answer,
        "generated_answer": answer,
        "reference_contexts": reference_contexts,
        "query_metadata": query_metadata,
        "answer_metadata": answer_metadata
    }

    if guardrails_blocked:
        metrics.update({
            "guardrail_input_assessment": guardrail_input_assessment,
            "guardrail_context_assessment": guardrail_context_assessment,
            "guardrail_output_assessment": guardrail_output_assessment,
            "guardrail_id": guardrail_id,
            "guardrail_blocked": guardrails_block_level
        })

    return metrics


def write_batch_to_metrics_db(batch_items: List[Dict], config_data, execution_id, experiment_id) -> None:
    """Write a batch of items to the appropriate database."""
    if config_data:
        write_batch_metrics_through_api(batch_items, config_data, execution_id, experiment_id)
    else:
        db_type = config.get_db_type()
        if db_type == "POSTGRESDB":
            write_batch_to_metrics_postgres(batch_items)
        elif db_type == "DYNAMODB":
            write_batch_to_metrics_dynamodb(batch_items)
        else:
            logger.error(f"Unsupported database type: {db_type}")
            raise ValueError(f"Unsupported database type: {db_type}")


def write_batch_to_metrics_dynamodb(batch_items: List[Dict]) -> None:
    """Write a batch of items to DynamoDB."""
    logger.info(f"Writing batch of {len(batch_items)} items to DynamoDB")
    db = DynamoDB(config.get_experiment_question_metrics_table())
    db.bulk_write(batch_items)


def write_batch_to_metrics_postgres(batch_items: List[Dict]) -> None:
    """Write a batch of items to PostgreSQL."""
    logger.info(f"Writing batch of {len(batch_items)} items to PostgreSQL")

    db = PostgresDB(
        dbname=config.get_postgress_db(),
        user=config.get_postgress_user(),
        password=config.get_postgress_password(),
        host=config.get_postgress_host(),
        port=config.get_postgress_port()
    )

    db.bulk_write(batch_items, config.get_experiment_question_metrics_table())

    db.close()

def write_batch_metrics_through_api(metrics_list: List[Dict[str, Any]], config_data: dict, project_uid, experiment_uid) -> None:
    """
    Transforms a list of metrics dictionaries into the required API payload format.

    Args:
        write_metrics_api:
        metrics_list (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
                                              represents a set of metrics for an item.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries formatted as required by the API.

    """
    required_metadata_keys = (
        "query_metadata",
        "answer_metadata",
        "guardrail_input_assessment",
        "guardrail_context_assessment",
        "guardrail_output_assessment",
    )
    transformed_payload_list = [
        {
            "input": [{"type": "question", "content": item_metrics.get("question", "")}],
            "expected": [{"type": "answer", "content": item_metrics.get("gt_answer", "")}],
            "actual": [{"type": "answer", "content": item_metrics.get("generated_answer", "")}],
            "metadata": [
                {"type": key, "content": item_metrics[key]}
                for key in required_metadata_keys
                if key in item_metrics
            ],
        }
        for item_metrics in metrics_list
    ]
    if not project_uid or not experiment_uid:
        logger.error("Missing projectUid or experimentUid in the first metrics item.")
    base_url = config_data.get("console", {}).get("url", "").rstrip("/")
    endpoint_path = config_data.get("console", {}).get("endpoints", {}).get("results", "")
    write_metrics_api = f"{base_url}{endpoint_path}?projectUid={project_uid}&experimentUid={experiment_uid}"
    headers_data = config_data.get("console", {}).get("headers", {})
    logger.info(f"Writing batch of {len(metrics_list)} items to API: {write_metrics_api}")
    url_storage = StorageProviderFactory.create_storage_provider(write_metrics_api)
    url_storage.write(write_metrics_api, transformed_payload_list, headers_data)
