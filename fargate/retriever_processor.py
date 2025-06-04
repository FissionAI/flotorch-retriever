from datetime import datetime
import json
from typing import Dict, List, Optional, Union
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
from flotorch_core.storage.db.vector.guardrails_vector_storage import GuardRailsVectorStorage
from flotorch_core.storage.db.vector.open_search import OpenSearchClient
from flotorch_core.inferencer.bedrock_inferencer import BedrockInferencer
from flotorch_core.storage.db.vector.vector_storage_factory import VectorStorageFactory
from flotorch_core.storage.storage_provider_factory import StorageProviderFactory

from flotorch_core.embedding.titanv2_embedding import TitanV2Embedding
from flotorch_core.embedding.titanv1_embedding import TitanV1Embedding
from flotorch_core.embedding.cohere_embedding import CohereEmbedding
from flotorch_core.embedding.bge_large_embedding import BGELargeEmbedding, BGEM3Embedding, GTEQwen2Embedding


logger = get_logger()
env_config_provider = EnvConfigProvider()
config = Config(env_config_provider)

class RetrieverProcessor(BaseFargateTaskProcessor):
    """
    Processor for retriever tasks in Fargate.
    """

    def process(self):
        logger.info("Starting retriever process.")
        try:
            exp_config_data = self.input_data

            n_shot_prompt_guide_obj = get_n_shot_prompt_guide_obj(exp_config_data.get("execution_id"))
            exp_config_data["n_shot_prompt_guide_obj"] = n_shot_prompt_guide_obj

            logger.info(f"Into retriever processor. Processing event: {json.dumps(exp_config_data)}")

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
                    write_batch_to_metrics_dynamodb(batch_items)
                    batch_items = []

            if len(batch_items) > 0:
                write_batch_to_metrics_dynamodb(batch_items)

            experiment_id = exp_config_data.get("experiment_id")
            update_tokens(experiment_id, total_input_tokens,total_output_tokens)
            output = {"status": "success", "message": "Retriever completed successfully."}
            self.send_task_success(output)
        except Exception as e:
            logger.error(f"Error during retriever process: {str(e)}")
            self.send_task_failure(str(e))

# get n shot prompt guide object
def get_n_shot_prompt_guide_obj(execution_id) -> Optional[Dict]:
    """
    Retrieves the n-shot prompt guide object from the dynamo db.
    """
    db = DynamoDB(config.get_execution_table_name())
    data = db.read({"id": execution_id})
    if data:
        n_shot_prompt_guide_obj = data.get("config", {}).get("n_shot_prompt_guide", None)
        return n_shot_prompt_guide_obj
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

def update_tokens(experiment_id, total_input_tokens, total_output_tokens):
    experiment_db = DynamoDB(config.get_experiment_table_name())
    token_fields = {
        "retrieval_input_tokens": total_input_tokens,
        "retrieval_output_tokens": total_output_tokens
    }
    if experiment_db.update({"id": experiment_id}, token_fields):
        logger.info(f"Updated tokens for experiment {experiment_id} successfully.")
    else:
        logger.error(f"Failed to update tokens for experiment {experiment_id}.")
    
    
def write_batch_to_metrics_dynamodb(batch_items: List[Dict]) -> None:
    """Write a batch of items to DynamoDB."""
    logger.info(f"Writing batch of {len(batch_items)} items to DynamoDB")
    db = DynamoDB(config.get_experiment_question_metrics_table())
    db.bulk_write(batch_items)
