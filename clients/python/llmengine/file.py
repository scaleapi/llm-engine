from llmengine.api_engine import DEFAULT_TIMEOUT, APIEngine
from llmengine.data_types import (
    DeleteFileResponse,
    GetFileContentResponse,
    GetFileResponse,
    ListFilesResponse,
    UploadFileResponse,
)


class File(APIEngine):
    """
    File API. This API is used to upload private files to Scale so that fine-tunes can access them for training and validation data.

    Functions are provided to upload, get, list, and delete files, as well as to get the contents of a file.
    """

    @classmethod
    def upload(cls, file_path: str) -> UploadFileResponse:
        """
        Uploads a local file to Scale.

        Args:
            file_path (`str`):
                The local path to the file.

        Returns:
            UploadFileResponse: an object that contains the ID of the uploaded file

        === "Uploading file in Python"
            ```python
            from llmengine import File

            response = File.upload("training_dataset.csv")

            print(response.json())
            ```

        === "Response in JSON"
            ```json
            {
                "id": "file-abc123"
            }
            ```
        """
        with open(file_path, "rb") as file:
            files = {"file": file}
            response = cls.post_file(
                resource_name="v1/files",
                files=files,
                timeout=DEFAULT_TIMEOUT,
            )
        return UploadFileResponse.parse_obj(response)

    @classmethod
    def get(cls, file_id: str) -> GetFileResponse:
        """
        Get filename and size of a file.

        Args:
            file_id (`str`):
                ID of the file

        Returns:
            GetFileResponse: an object that contains the ID, filename, and size of the requested file

        === "Getting metadata about file in Python"
            ```python
            from llmengine import File

            response = File.get(
                file_id="file-abc123",
            )

            print(response.json())
            ```

        === "Response in JSON"
            ```json
            {
                "id": "file-abc123",
                "filename": "training_dataset.csv",
                "size": 100
            }
            ```
        """
        response = cls._get(f"v1/files/{file_id}", timeout=DEFAULT_TIMEOUT)
        return GetFileResponse.parse_obj(response)

    @classmethod
    def list(cls) -> ListFilesResponse:
        """
        List all files, with information about their filenames and sizes.

        Returns:
            ListFilesResponse: an object that contains a list of all files and their filenames and sizes

        === "Listing files in Python"
            ```python
            from llmengine import File

            response = File.list()
            print(response.json())
            ```

        === "Response in JSON"
            ```json
            {
                "files": [
                    {
                        "id": "file-abc123",
                        "filename": "training_dataset.csv",
                        "size": 100
                    },
                    {
                        "id": "file-def456",
                        "filename": "validation_dataset.csv",
                        "size": 50
                    }
                ]
            }
            ```
        """
        response = cls._get("v1/files", timeout=30)
        return ListFilesResponse.parse_obj(response)

    @classmethod
    def delete(cls, file_id: str) -> DeleteFileResponse:
        """
        Deletes a file.

        Args:
            file_id (`str`):
                ID of the file

        Returns:
            DeleteFileResponse: an object that contains whether the deletion was successful

        === "Deleting file in Python"
            ```python
            from llmengine import File

            response = File.delete(file_id="file-abc123")
            print(response.json())
            ```

        === "Response in JSON"
            ```json
            {
                "deleted": true
            }
            ```
        """
        response = cls._delete(
            f"v1/files/{file_id}",
            timeout=DEFAULT_TIMEOUT,
        )
        return DeleteFileResponse.parse_obj(response)

    @classmethod
    def get_content(cls, file_id: str) -> GetFileContentResponse:
        """
        Get contents of a file.

        Args:
            file_id (`str`):
                ID of the file

        Returns:
            GetFileContentResponse: an object that contains the ID and content of the file

        === "Getting file content in Python"
            ```python
            from llmengine import File

            response = File.get_content(file_id="file-abc123")
            print(response.json())
            ```

        === "Response in JSON"
            ```json
            {
                "id": "file-abc123",
                "content": "Hello world!"
            }
            ```
        """
        response = cls._get(
            f"v1/files/{file_id}/content",
            timeout=DEFAULT_TIMEOUT,
        )
        return GetFileContentResponse.parse_obj(response)
