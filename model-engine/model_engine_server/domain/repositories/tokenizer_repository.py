from abc import ABC, abstractmethod

from transformers import AutoTokenizer


class TokenizerRepository(ABC):
    @abstractmethod
    def load_tokenizer(self, model_name: str) -> AutoTokenizer:
        """
        Loads a tokenizer from a model name.

        Args:
            model_name: The model name to load the tokenizer for.

        Returns:
            A tokenizer.
        """
        pass
