import pytest
from datetime import datetime
from uuid import UUID, uuid4
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from quivr_core.rag.entities.chat import ChatHistory


class TestChatHistory:
    """Test suite for ChatHistory class"""

    def test_init(self):
        """Test ChatHistory initialization"""
        chat_id = uuid4()
        brain_id = uuid4()

        chat_history = ChatHistory(chat_id=chat_id, brain_id=brain_id)

        assert chat_history.id == chat_id
        assert chat_history.brain_id == brain_id
        assert len(chat_history._msgs) == 0
        assert len(chat_history) == 0

    def test_init_with_none_brain_id(self):
        """Test ChatHistory initialization with None brain_id"""
        chat_id = uuid4()

        chat_history = ChatHistory(chat_id=chat_id, brain_id=None)

        assert chat_history.id == chat_id
        assert chat_history.brain_id is None
        assert len(chat_history._msgs) == 0

    def test_append_human_message(self):
        """Test appending a human message to chat history"""
        chat_id = uuid4()
        brain_id = uuid4()
        chat_history = ChatHistory(chat_id=chat_id, brain_id=brain_id)

        human_msg = HumanMessage(content="Hello, how are you?")
        metadata = {"user_id": "123", "timestamp": "2023-01-01"}

        with patch("quivr_core.rag.entities.chat.datetime") as mock_datetime:
            mock_time = datetime(2023, 1, 1, 12, 0, 0)
            mock_datetime.now.return_value = mock_time

            chat_history.append(human_msg, metadata)

        assert len(chat_history) == 1
        assert len(chat_history._msgs) == 1

        stored_msg = chat_history._msgs[0]
        assert stored_msg.chat_id == chat_id
        assert stored_msg.brain_id == brain_id
        assert stored_msg.msg == human_msg
        assert stored_msg.metadata == metadata
        assert stored_msg.message_time == mock_time

    def test_append_ai_message(self):
        """Test appending an AI message to chat history"""
        chat_id = uuid4()
        brain_id = uuid4()
        chat_history = ChatHistory(chat_id=chat_id, brain_id=brain_id)

        ai_msg = AIMessage(content="I'm doing well, thank you!")

        chat_history.append(ai_msg)

        assert len(chat_history) == 1
        stored_msg = chat_history._msgs[0]
        assert stored_msg.msg == ai_msg
        assert stored_msg.metadata == {}

    def test_append_multiple_messages(self):
        """Test appending multiple messages to chat history"""
        chat_id = uuid4()
        brain_id = uuid4()
        chat_history = ChatHistory(chat_id=chat_id, brain_id=brain_id)

        human_msg1 = HumanMessage(content="Hello")
        ai_msg1 = AIMessage(content="Hi there!")
        human_msg2 = HumanMessage(content="How are you?")
        ai_msg2 = AIMessage(content="I'm doing well!")

        chat_history.append(human_msg1)
        chat_history.append(ai_msg1)
        chat_history.append(human_msg2)
        chat_history.append(ai_msg2)

        assert len(chat_history) == 4
        assert chat_history._msgs[0].msg == human_msg1
        assert chat_history._msgs[1].msg == ai_msg1
        assert chat_history._msgs[2].msg == human_msg2
        assert chat_history._msgs[3].msg == ai_msg2

    def test_get_chat_history_chronological(self):
        """Test getting chat history in chronological order"""
        chat_id = uuid4()
        brain_id = uuid4()
        chat_history = ChatHistory(chat_id=chat_id, brain_id=brain_id)

        # Add messages with different timestamps
        with patch("quivr_core.rag.entities.chat.datetime") as mock_datetime:
            # First message
            mock_datetime.now.return_value = datetime(2023, 1, 1, 10, 0, 0)
            chat_history.append(HumanMessage(content="First message"))

            # Second message
            mock_datetime.now.return_value = datetime(2023, 1, 1, 11, 0, 0)
            chat_history.append(AIMessage(content="Second message"))

            # Third message
            mock_datetime.now.return_value = datetime(2023, 1, 1, 9, 0, 0)
            chat_history.append(HumanMessage(content="Third message"))

        history = chat_history.get_chat_history(newest_first=False)

        # Should be sorted by message_time, oldest first
        assert len(history) == 3
        assert history[0].msg.content == "Third message"  # 9:00
        assert history[1].msg.content == "First message"  # 10:00
        assert history[2].msg.content == "Second message"  # 11:00

    def test_get_chat_history_reverse_chronological(self):
        """Test getting chat history in reverse chronological order"""
        chat_id = uuid4()
        brain_id = uuid4()
        chat_history = ChatHistory(chat_id=chat_id, brain_id=brain_id)

        # Add messages with different timestamps
        with patch("quivr_core.rag.entities.chat.datetime") as mock_datetime:
            # First message
            mock_datetime.now.return_value = datetime(2023, 1, 1, 10, 0, 0)
            chat_history.append(HumanMessage(content="First message"))

            # Second message
            mock_datetime.now.return_value = datetime(2023, 1, 1, 11, 0, 0)
            chat_history.append(AIMessage(content="Second message"))

            # Third message
            mock_datetime.now.return_value = datetime(2023, 1, 1, 9, 0, 0)
            chat_history.append(HumanMessage(content="Third message"))

        history = chat_history.get_chat_history(newest_first=True)

        # Should be sorted by message_time, newest first
        assert len(history) == 3
        assert history[0].msg.content == "Second message"  # 11:00
        assert history[1].msg.content == "First message"  # 10:00
        assert history[2].msg.content == "Third message"  # 9:00

    def test_get_chat_history_empty(self):
        """Test getting chat history when empty"""
        chat_id = uuid4()
        brain_id = uuid4()
        chat_history = ChatHistory(chat_id=chat_id, brain_id=brain_id)

        history = chat_history.get_chat_history()
        assert history == []

        history_reverse = chat_history.get_chat_history(newest_first=True)
        assert history_reverse == []

    def test_iter_pairs_valid(self):
        """Test iterating over chat history in pairs"""
        chat_id = uuid4()
        brain_id = uuid4()
        chat_history = ChatHistory(chat_id=chat_id, brain_id=brain_id)

        # Add messages in the correct order (newest first for iteration)
        with patch("quivr_core.rag.entities.chat.datetime") as mock_datetime:
            # Add in chronological order
            mock_datetime.now.return_value = datetime(2023, 1, 1, 10, 0, 0)
            chat_history.append(HumanMessage(content="Question 1"))

            mock_datetime.now.return_value = datetime(2023, 1, 1, 10, 1, 0)
            chat_history.append(AIMessage(content="Answer 1"))

            mock_datetime.now.return_value = datetime(2023, 1, 1, 10, 2, 0)
            chat_history.append(HumanMessage(content="Question 2"))

            mock_datetime.now.return_value = datetime(2023, 1, 1, 10, 3, 0)
            chat_history.append(AIMessage(content="Answer 2"))

        pairs = list(chat_history.iter_pairs())

        assert len(pairs) == 2
        # Should iterate newest first, so pairs are (Human, AI)
        human_msg1, ai_msg1 = pairs[0]
        assert human_msg1.content == "Question 2"
        assert ai_msg1.content == "Answer 2"

        human_msg2, ai_msg2 = pairs[1]
        assert human_msg2.content == "Question 1"
        assert ai_msg2.content == "Answer 1"

    def test_iter_pairs_invalid_order(self):
        """Test iter_pairs with invalid message order"""
        chat_id = uuid4()
        brain_id = uuid4()
        chat_history = ChatHistory(chat_id=chat_id, brain_id=brain_id)

        # Add messages in wrong order (AI first, then Human)
        with patch("quivr_core.rag.entities.chat.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 1, 1, 10, 0, 0)
            chat_history.append(AIMessage(content="Answer first"))

            mock_datetime.now.return_value = datetime(2023, 1, 1, 10, 1, 0)
            chat_history.append(HumanMessage(content="Question second"))

        with pytest.raises(AssertionError):
            list(chat_history.iter_pairs())

    def test_iter_pairs_empty(self):
        """Test iter_pairs with empty chat history"""
        chat_id = uuid4()
        brain_id = uuid4()
        chat_history = ChatHistory(chat_id=chat_id, brain_id=brain_id)

        pairs = list(chat_history.iter_pairs())
        assert pairs == []

    def test_iter_pairs_odd_number_messages(self):
        """Test iter_pairs with odd number of messages"""
        chat_id = uuid4()
        brain_id = uuid4()
        chat_history = ChatHistory(chat_id=chat_id, brain_id=brain_id)

        # Add only one message
        chat_history.append(HumanMessage(content="Single message"))

        pairs = list(chat_history.iter_pairs())
        assert pairs == []  # No pairs can be formed

    def test_to_list(self):
        """Test converting chat history to list of messages"""
        chat_id = uuid4()
        brain_id = uuid4()
        chat_history = ChatHistory(chat_id=chat_id, brain_id=brain_id)

        human_msg = HumanMessage(content="Hello")
        ai_msg = AIMessage(content="Hi there!")

        chat_history.append(human_msg)
        chat_history.append(ai_msg)

        msg_list = chat_history.to_list()

        assert len(msg_list) == 2
        assert msg_list[0] == human_msg
        assert msg_list[1] == ai_msg
        assert isinstance(msg_list[0], HumanMessage)
        assert isinstance(msg_list[1], AIMessage)

    def test_to_list_empty(self):
        """Test converting empty chat history to list"""
        chat_id = uuid4()
        brain_id = uuid4()
        chat_history = ChatHistory(chat_id=chat_id, brain_id=brain_id)

        msg_list = chat_history.to_list()
        assert msg_list == []

    def test_len_operator(self):
        """Test __len__ method"""
        chat_id = uuid4()
        brain_id = uuid4()
        chat_history = ChatHistory(chat_id=chat_id, brain_id=brain_id)

        assert len(chat_history) == 0

        chat_history.append(HumanMessage(content="Test"))
        assert len(chat_history) == 1

        chat_history.append(AIMessage(content="Response"))
        assert len(chat_history) == 2

    def test_message_id_uniqueness(self):
        """Test that each message gets a unique ID"""
        chat_id = uuid4()
        brain_id = uuid4()
        chat_history = ChatHistory(chat_id=chat_id, brain_id=brain_id)

        chat_history.append(HumanMessage(content="Message 1"))
        chat_history.append(HumanMessage(content="Message 2"))

        msg1_id = chat_history._msgs[0].message_id
        msg2_id = chat_history._msgs[1].message_id

        assert msg1_id != msg2_id
        assert isinstance(msg1_id, UUID)
        assert isinstance(msg2_id, UUID)

    def test_metadata_preservation(self):
        """Test that metadata is properly preserved"""
        chat_id = uuid4()
        brain_id = uuid4()
        chat_history = ChatHistory(chat_id=chat_id, brain_id=brain_id)

        metadata = {
            "user_id": "test_user",
            "session_id": "test_session",
            "custom_data": {"key": "value"},
        }

        chat_history.append(HumanMessage(content="Test"), metadata)

        stored_metadata = chat_history._msgs[0].metadata
        assert stored_metadata == metadata
        assert stored_metadata["user_id"] == "test_user"
        assert stored_metadata["custom_data"]["key"] == "value"
