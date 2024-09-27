"""
DTOs for Files API.
"""

from typing import List

from model_engine_server.common.pydantic_types import BaseModel, Field


class UploadFileResponse(BaseModel):
    """Response object for uploading a file."""

    id: str = Field(..., description="ID of the uploaded file.")
    """ID of the uploaded file."""


class GetFileResponse(BaseModel):
    """Response object for retrieving a file."""

    id: str = Field(..., description="ID of the requested file.")
    """ID of the requested file."""
    filename: str = Field(..., description="File name.")
    """File name."""
    size: int = Field(..., description="Length of the file, in characters.")
    """Length of the file, in characters."""


class ListFilesResponse(BaseModel):
    """Response object for listing files."""

    files: List[GetFileResponse] = Field(..., description="List of file IDs, names, and sizes.")
    """List of file IDs, names, and sizes."""


class DeleteFileResponse(BaseModel):
    """Response object for deleting a file."""

    deleted: bool = Field(..., description="Whether deletion was successful.")
    """Whether deletion was successful."""


class GetFileContentResponse(BaseModel):
    """Response object for retrieving a file's content."""

    id: str = Field(..., description="ID of the requested file.")
    """ID of the requested file."""
    content: str = Field(..., description="File content.")
    """File content."""
