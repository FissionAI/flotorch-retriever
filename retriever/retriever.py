from dataclasses import dataclass, field
from typing import Any, Dict, List
from flotorch_core.reader.json_reader import JSONReader
from flotorch_core.embedding.embedding import BaseEmbedding
from flotorch_core.chunking.chunking import Chunk
from flotorch_core.rerank.rerank import BedrockReranker
from flotorch_core.storage.db.vector.vector_storage import VectorStorage
from pydantic import BaseModel
from flotorch_core.inferencer.inferencer import BaseInferencer

class Question(BaseModel):
    question: str
    answer: str

    def get_chunk(self) -> Chunk:
        return Chunk(data=self.question)
    

@dataclass
class RetrieverItem:
    question: str = field(default_factory=None)
    answer: str = field(default_factory=None)
    guardrails_output_assessment:List = field(default_factory=None)
    guardrails_context_assessment: List = field(default_factory=None)
    guardrails_input_assessment: List = field(default_factory=None)
    guardrails_blocked: bool = field(default_factory=False)
    guardrails_block_level: str = field(default_factory="")
    answer_metadata: Dict[str, Any] = field(default_factory={})
    query_metadata: Dict[str, Any] = field(default_factory={})
    reference_contexts: List[str] = field(default_factory=[])
    gt_answer: str = field(default_factory=None)

    def to_json(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "guardrails_output_assessment": self.guardrails_output_assessment,
            "guardrails_context_assessment": self.guardrails_context_assessment,
            "guardrails_input_assessment": self.guardrails_input_assessment,
            "guardrails_blocked": self.guardrails_blocked,
            "guardrails_block_level": self.guardrails_block_level,
            "answer_metadata": self.answer_metadata,
            "query_metadata": self.query_metadata,
            "reference_contexts": self.reference_contexts,
            "gt_answer": self.gt_answer
        }

class Retriever:
    def __init__(self, json_reader: JSONReader, embedding: BaseEmbedding, vector_storage: VectorStorage, inferencer: BaseInferencer, reranker: BedrockReranker) -> None:
        self.json_reader = json_reader
        self.embedding = embedding
        self.vector_storage = vector_storage
        self.inferencer = inferencer
        self.reranker = reranker

    def retrieve(self, path: str, query: str, knn: int, hierarchical: bool = False):
        result = []
        total_input_tokens = 0
        total_output_tokens = 0
        questions_list = self.json_reader.read_as_model(path, Question)
        for question in questions_list:
            question_chunk = question.get_chunk()
            response = self.vector_storage.search(question_chunk, knn, hierarchical)
            vector_response = response.to_json()['result']
            
            if response.status:
                metadata, answer = self.inferencer.generate_text(question.question, vector_response)
                guardrail_blocked = metadata['guardrail_blocked'] if 'guardrail_blocked' in metadata else False
                if guardrail_blocked:

                    answer_metadata = {}
                else:
                    answer_metadata = metadata
                    total_input_tokens += metadata['inputTokens']
                    total_output_tokens += metadata['outputTokens']
            else:
                answer = response.metadata['guardrail_output']
                metadata = {}
                answer_metadata = {}
                guardrail_blocked = response.metadata['guardrail_blocked'] if 'guardrail_blocked' in response.metadata else False

            result.append(RetrieverItem(
                question=question.question,
                answer=answer,
                guardrails_output_assessment=metadata['guardrail_output_assessment'] if 'guardrail_output_assessment' in metadata else None,
                guardrails_context_assessment=response.metadata['guardrail_context_assessment'] if 'guardrail_context_assessment' in response.metadata else None,
                guardrails_input_assessment=response.metadata['guardrail_input_assessment'] if 'guardrail_input_assessment' in response.metadata else None,
                guardrails_blocked=guardrail_blocked,
                guardrails_block_level=response.metadata['block_level'] if 'block_level' in response.metadata else "",
                answer_metadata=answer_metadata,
                reference_contexts=[res['text'] for res in vector_response] if vector_response else [],
                gt_answer=question.answer,
                query_metadata=response.metadata['embedding_metadata'].to_json() if 'embedding_metadata' in response.metadata else None
            ))

        return result, total_input_tokens, total_output_tokens
            
            
