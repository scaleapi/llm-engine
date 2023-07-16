from unittest.mock import MagicMock, patch

import pytest

from llm_engine_server.db.models.common.record import Record


@pytest.fixture
def mock_session():
    return MagicMock()


@pytest.fixture
def mock_query():
    return MagicMock()


class TestRecord:
    """
    Test the Record class.
    """

    def test_create(self, mock_session):
        item = MagicMock()
        Record.create(session=mock_session, record=item)
        mock_session.add.assert_called_once_with(item)
        mock_session.commit.assert_called_once_with()

    @patch("llm_engine_server.db.models.common.record.select")
    def test_select_all(self, mock_select, mock_session, mock_query):
        mock_query.to_sqlalchemy_query.return_value = {"id": "123", "name": "test"}
        mock_select_obj = MagicMock()
        mock_select.return_value = mock_select_obj
        Record.select_all(session=mock_session, query=mock_query)
        mock_select.assert_called_once_with(Record)
        mock_select_obj.filter_by.assert_called_once_with(id="123", name="test")
        mock_session.execute.assert_called_once_with(mock_select_obj.filter_by.return_value)
        mock_session.execute.return_value.scalars.assert_called_once_with()
        mock_session.execute.return_value.scalars.return_value.all.assert_called_once_with()

    @patch("llm_engine_server.db.models.common.record.select")
    def test_select_by_id(self, mock_select, mock_session):
        mock_select_obj = MagicMock()
        mock_select.return_value = mock_select_obj
        Record.select_by_id(session=mock_session, record_id="123")
        mock_select.assert_called_once_with(Record)
        mock_select_obj.filter_by.assert_called_once_with(id="123")
        mock_session.execute.assert_called_once_with(mock_select_obj.filter_by.return_value)
        mock_session.execute.return_value.scalar_one_or_none.assert_called_once_with()

    @patch("llm_engine_server.db.models.common.record.select")
    def test_update(self, mock_select, mock_session, mock_query):
        mock_select_obj = MagicMock()
        mock_select.return_value = mock_select_obj
        mock_query.to_sqlalchemy_query.return_value = {"name": "test"}
        item = MagicMock()
        mock_session.execute.return_value.scalar_one_or_none.return_value = item
        Record.update(session=mock_session, record_id="123", query=mock_query)
        mock_select.assert_called_once_with(Record)
        mock_select_obj.filter_by.assert_called_once_with(id="123")
        mock_session.execute.assert_called_once_with(mock_select_obj.filter_by.return_value)
        mock_session.execute.return_value.scalar_one_or_none.assert_called_once_with()
        item.name = "test"
        mock_session.commit.assert_called_once_with()

    def test_delete(self, mock_session):
        item = MagicMock()
        Record.delete(session=mock_session, record=item)
        mock_session.delete.assert_called_once_with(item)
        mock_session.commit.assert_called_once_with()
