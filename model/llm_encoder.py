from typing import Any, Dict, List, Optional

from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, Extra, Field
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL_NAME = "lmsys/vicuna-13b-v1.5-16k"

class LLMBasedEmbeddings(BaseModel, Embeddings):
    model_name: str = DEFAULT_MODEL_NAME
    """Model name to use."""

    def __init__(self, encodingmodel=model_name, **kwargs):
        super().__init__(**kwargs)
        """Initialize the encoding model."""
        self.device = kwargs.get('device', 'cpu')
        self.aggregation = kwargs.get('aggr', 'mean')
        self.model = AutoModel.from_pretrained(encodingmodel,
                                               use_auth_token=kwargs.get('token', ''),
                                               output_hidden_states=True).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(encodingmodel,
                                                       token=kwargs.get('token', ''),
                                                       device=self.device)

    class Config:
        extra = "allow"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a LLM model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """

        texts = list(map(lambda x: x.replace("\n", " "), texts))
        if self.aggregation=='max':
            embeddings = [self.model(
                self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device))
                          ["last_hidden_state"]
                          .max(axis=1).values[0].detach().numpy() for text in texts]

        else:
            embeddings = [self.model(
                self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device))
                          ["last_hidden_state"]
                          .mean(axis=[0, 1]).detach().numpy() for text in texts]

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a LLM model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]