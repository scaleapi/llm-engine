from abc import ABC, abstractmethod
from typing import Any, Dict, List


class LLMArtifactGateway(ABC):
    """
    Abstract Base Class for interacting with llm artifacts.
    """

    @abstractmethod
    def list_files(self, path: str, **kwargs) -> List[str]:
        """
        Gets a list of files from a given path.

        Args:
            path (str): path to list files
        """
        pass

    @abstractmethod
    def download_files(self, path: str, target_path: str, overwrite=False, **kwargs) -> List[str]:
        """
        Download files from a given path to a target path.

        Args:
            path (str): path to list files
            target_path (str): local path to download files
            overwrite (bool): whether to overwrite existing local files
        """
        pass

    @abstractmethod
    def get_model_weights_urls(self, owner: str, model_name: str, **kwargs) -> List[str]:
        """
        Gets a list of URLs for all files associated with a given model.

        Args:
            owner (str): owner of the model
            model_name (str): name of the model
        """
        pass

    @abstractmethod
    def get_model_config(self, path: str, **kwargs) -> Dict[str, Any]:
        """
        Gets the model config from the model files live at given folder.

        Args:
            path (str): path to model files
        """
        pass

    @abstractmethod
    def get_tokenizer_config(self, path: str, **kwargs) -> Dict[str, Any]:
        """
        Gets the tokenizer config from the model files live at given folder.

        Args:
            path (str): path to tokenizer_config.json
        """
        pass
