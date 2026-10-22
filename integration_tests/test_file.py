import pytest

from .rest_api_utils import upload_file


def test_upload_file_is_deprecated() -> None:
    user = "62bc820451dbea002b1c5421"  # CDS needs proper user ID

    with pytest.raises(ValueError, match="deprecated"):
        upload_file(open(__file__, "rb"), user)
