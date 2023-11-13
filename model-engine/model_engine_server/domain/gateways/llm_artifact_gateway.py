from abc import ABC, abstractmethod
from typing import List


class LLMArtifactGateway(ABC):
    """
    Abstract Base Class for interacting with llm artifacts.
    """

    @abstractmethod
    def list_files(self, path: str, **kwargs) -> List[str]:
        """
        Gets a list of files from a given path.

        Args:
            path (str): S3 path to list files for. can either be a complete path or prefix
        """
        pass

    @abstractmethod
    def download_files(self, path: str, target_path: str, overwrite=False, **kwargs) -> List[str]:
        """
        Download files from a given path to a target path.

        Args:
            path (str): S3 path to list files for. can either be a complete path or prefix
            target_path (str): local file dir path to download files to. can either be a complete path or prefix
            overwrite (bool): whether to overwrite existing files
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
