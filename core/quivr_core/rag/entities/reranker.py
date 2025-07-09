from enum import Enum
from pydantic_settings import BaseSettings


class DefaultRerankers(str, Enum):
    COHERE = "cohere"
    JINA = "jina"

    @property
    def default_model(self) -> str:
        # Mapping of suppliers to their default models
        return {
            self.COHERE: "rerank-v3.5",
            self.JINA: "jina-reranker-v2-base-multilingual",
        }[self]


class RerankerConfig(BaseSettings):
    supplier: DefaultRerankers | None = None
    model: str | None = None
    top_n: int = 5  # Number of chunks returned by the re-ranker
    relevance_score_threshold: float | None = None
    relevance_score_key: str = "relevance_score"

    cohere_api_key: str | None = None
    jina_api_key: str | None = None

    def __init__(self, **data):
        super().__init__(**data)
        self.validate_model()

    @property
    def api_key(self) -> str | None:
        """Get the appropriate API key based on the supplier"""
        if not self.supplier:
            return None

        supplier_key_map = {
            DefaultRerankers.COHERE: self.cohere_api_key,
            DefaultRerankers.JINA: self.jina_api_key,
        }

        api_key = supplier_key_map.get(self.supplier)
        if api_key is None and self.supplier:
            raise ValueError(f"The API key for supplier '{self.supplier}' is not set. ")

        return api_key

    def validate_model(self):
        # If model is not provided, get default model based on supplier
        if self.model is None and self.supplier is not None:
            self.model = self.supplier.default_model
