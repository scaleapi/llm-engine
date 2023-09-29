"""Utilities for handling parsing of files."""


def exclude_safetensors_or_bin(model_files):
    """
    This function is used to determine whether to exclude "*.safetensors" or "*.bin" files
    based on which file type is present more often in the checkpoint folder.
    """
    exclude_str = ""
    if len([f for f in model_files if f.endswith(".safetensors")]) > len(
        [f for f in model_files if f.endswith(".bin")]
    ):
        exclude_str = "*.bin"
    else:
        exclude_str = "*.safetensors"
    return exclude_str
