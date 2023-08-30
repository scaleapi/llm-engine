from .rest_api_utils import (
    upload_file,
    get_file_by_id,
    list_files,
    delete_file_by_id,
    get_file_content_by_id,
)

def test_files() -> None:
    user = "62bc820451dbea002b1c5421"  # CDS needs proper user ID

    upload_response = upload_file(open(__file__, "rb"), user)
    file_id = upload_response["id"]

    content = get_file_content_by_id(file_id, user)
    assert content["id"] == file_id
    assert content["content"]

    get_response = get_file_by_id(file_id, user)
    assert get_response["id"] == file_id
    assert get_response["filename"] == "test_file.py"

    list_response = list_files(user)
    assert len(list_response["files"]) > 0

    delete_response = delete_file_by_id(file_id, user)
    assert delete_response["deleted"]
