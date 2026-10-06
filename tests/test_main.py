from unittest.mock import MagicMock

import fixtures
import llm
import pytest
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Caption


def test_no_interaction() -> None:
    """App renders its initial empty state before any prompt is submitted."""
    at: AppTest = AppTest.from_file("../aiviz/main.py", default_timeout=30).run()
    assert not at.exception

    # session state initialized with no tokens
    assert at.session_state.tokens == []

    # llm output box exists and holds only the placeholder caption
    output_box = at.columns[0].children[0]
    assert output_box.type == "flex_container"
    assert len(output_box.children) == 1
    placeholder = output_box.children[0]
    assert isinstance(placeholder, Caption)
    assert placeholder.value == "Response will appear here."

    # prompt input exists with a non-empty placeholder
    assert len(at.chat_input) == 1
    assert at.chat_input[0].placeholder

    # visualization tabs are rendered
    assert at.tabs


def test_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    """Submitting a prompt runs the model and renders the completed response."""
    monkeypatch.delenv("USE_MOCK_LLM", raising=False)

    # patch model
    tokens = fixtures.mock_tokens()
    model = MagicMock(spec=llm.ProfiledSmolLM, MODEL_ID="fake-model")
    model.run.return_value = tokens

    at: AppTest = AppTest.from_file("../aiviz/main.py", default_timeout=30)
    at.session_state["model"] = model
    at.run()

    # set input prompt
    at.chat_input[0].set_value("Hello, world!").run()
    assert not at.exception, at.exception[0].message

    model.run.assert_called_once_with(user_input="Hello, world!")

    output_box = at.columns[0].children[0]

    # check output messages (user message, llm response timeline, llm response)
    assert len(output_box.children) == 3
    user, timeline, response = output_box.chat_message
    del timeline
    assert user.markdown[0].value == "Hello, world!"
    assert response.markdown[0].value == "".join(token.text for token in tokens)
