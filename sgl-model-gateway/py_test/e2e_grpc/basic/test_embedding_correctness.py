"""
gRPC Router E2E Test - Embedding Correctness

Test that embeddings from the gRPC router match HuggingFace reference embeddings.
Validates numerical correctness including tokenization (BOS/EOS handling) and inference.
"""

import logging
import sys
import unittest
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as F

_TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(_TEST_DIR.parent))
from fixtures import popen_launch_workers_and_router
from util import (
    DEFAULT_EMBEDDING_MODEL_PATH,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    kill_process_tree,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Test data for semantic similarity checks
SEMANTIC_TEST_SETS = [
    [
        "Hello world",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
    ],
    [
        "SGLang provides fast and efficient LLM serving.",
        "Natural language processing enables computers to understand text.",
    ],
]

# Test data for relevance score comparison
RELEVANCE_TEST_DATA = {
    "sample_query": "What is machine learning?",
    "sample_reference": [
        {"body": "Machine learning is a branch of artificial intelligence that enables systems to learn from data."},
        {"body": "The weather today is sunny with a high of 75 degrees."},
        {"body": "Deep learning uses neural networks with multiple layers to model complex patterns."},
    ],
}


def get_openai_embeddings(
    texts: Union[str, List[str]], config: Dict
) -> List[List[float]]:
    """Get embeddings from the gateway via OpenAI-compatible API."""
    import openai

    client = openai.Client(api_key=config["api_key"], base_url=config["base_url"])

    if isinstance(texts, str):
        texts = [texts]

    embeddings = []
    for text in texts:
        response = client.embeddings.create(
            model=config["model_name"],
            input=text,
        )
        embeddings.append(response.data[0].embedding)

    return embeddings


def get_hf_embeddings(texts: Union[str, List[str]], model_path: str) -> torch.Tensor:
    """Get embeddings using HuggingFace transformers with mean pooling."""
    from transformers import AutoModel, AutoTokenizer

    if isinstance(texts, str):
        texts = [texts]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]

            # Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            embedding = sum_embeddings / sum_mask

            # Normalize
            embedding = F.normalize(embedding, p=2, dim=1)
            embeddings.append(embedding.cpu())

    return torch.cat(embeddings, dim=0)


def get_hf_st_embeddings(texts: Union[str, List[str]], model_path: str) -> np.ndarray:
    """Get embeddings using sentence-transformers library."""
    from sentence_transformers import SentenceTransformer

    if isinstance(texts, str):
        texts = [texts]

    model = SentenceTransformer(model_path, trust_remote_code=True)
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings


def get_input_texts(test_json: Dict) -> List[str]:
    """Extract document bodies from test JSON."""
    return [doc["body"] for doc in test_json["sample_reference"]]


def compare_embeddings(
    embeddings1: List[List[float]], embeddings2: List[List[float]]
) -> List[float]:
    """Compare two sets of embeddings using cosine similarity."""
    logging.info("Comparing embeddings")
    similarities = [
        F.cosine_similarity(torch.tensor(e1), torch.tensor(e2), dim=0).item()
        for e1, e2 in zip(embeddings1, embeddings2)
    ]
    return similarities


class TestEmbeddingCorrectness(CustomTestCase):
    """Test embedding correctness by comparing gateway output against HuggingFace reference."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_EMBEDDING_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        # Launch workers with --is-embedding flag
        cls.cluster = popen_launch_workers_and_router(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            num_workers=1,
            tp_size=1,
            policy="round_robin",
            api_key=cls.api_key,
            worker_args=["--is-embedding"],
        )

        cls.config = {
            "server_engine": "sgl-model-gateway",
            "base_url": cls.base_url + "/v1",
            "model_name": cls.model,
            "model_path": cls.model,
            "api_key": cls.api_key,
        }

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.cluster["router"].pid)
        for worker in cls.cluster.get("workers", []):
            kill_process_tree(worker.pid)

    def test_semantic_similarity(self, tolerance: float = 1e-2):
        """Check if gateway and HF embeddings give similar results."""
        for i, input_texts in enumerate(SEMANTIC_TEST_SETS):
            logging.info(f"Processing semantic similarity test set {i + 1}")

            embedding_openai = get_openai_embeddings(input_texts, self.config)
            embedding_hf = get_hf_embeddings(input_texts, self.config["model_path"]).detach().tolist()

            logging.info(f'Comparing {self.config["server_engine"]} and HF embeddings')
            similarities = compare_embeddings(embedding_openai, embedding_hf)

            logging.info(f"Similarities between embeddings: {similarities}")

            # Verify similarities
            for sim in similarities:
                self.assertLess(
                    abs(sim - 1.0),
                    tolerance,
                    f"Similarity {sim} is not close to 1"
                )

            logging.info(f"Semantic similarity test set {i + 1} passed\n")

    def test_relevance_scores(self, tolerance: float = 5e-2):
        """Compare relevance scores between gateway and HF implementations."""
        logging.info(f'Comparing relevance scores between {self.config["server_engine"]} and HF')

        # Format query with instruction (for e5-mistral)
        query = f"Instruct: Given a search query, retrieve relevant passages that answer the query\nQuery: {RELEVANCE_TEST_DATA['sample_query']}"
        docs = get_input_texts(RELEVANCE_TEST_DATA)

        # Get gateway scores
        query_embeddings_openai = get_openai_embeddings(query, self.config)
        docs_embeddings_openai = get_openai_embeddings(docs, self.config)
        scores_openai = (
            np.array(query_embeddings_openai) @ np.array(docs_embeddings_openai).T
        ) * 100

        # Get HF scores using sentence-transformers
        query_embeddings_hf = get_hf_st_embeddings(query, self.config["model_path"])
        docs_embeddings_hf = get_hf_st_embeddings(docs, self.config["model_path"])
        scores_hf = (query_embeddings_hf @ docs_embeddings_hf.T) * 100

        logging.info(f'{self.config["server_engine"]} relevance scores: {scores_openai}')
        logging.info(f"HF relevance scores: {scores_hf}")

        self.assertTrue(
            np.allclose(scores_openai, scores_hf, atol=tolerance),
            f'Scores differ beyond tolerance: \n{self.config["server_engine"]}: {scores_openai}\nHF: {scores_hf}'
        )

        logging.info("Relevance scores comparison completed successfully")


if __name__ == "__main__":
    unittest.main()
