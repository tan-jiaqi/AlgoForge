import pytest
from pydantic import BaseModel

from factory.llm import CassetteLLM, LLMError, LLMSettings, MockLLM, _redact, extract_json, make_llm


class Answer(BaseModel):
    value: int


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('Sure! {"a": 3} hope this helps') == {"a": 3}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_structured_retries_once_with_validation_error():
    llm = MockLLM().queue("t", '{"value": "oops"}', '{"value": 7}')
    assert llm.structured([{"role": "user", "content": "q"}], Answer, tag="t").value == 7
    assert len(llm.calls) == 2
    assert "invalid" in llm.calls[1].messages[-1]["content"]


def test_structured_gives_up_after_retry():
    llm = MockLLM().queue("t", "bad", "still bad")
    with pytest.raises(LLMError):
        llm.structured([{"role": "user", "content": "q"}], Answer, tag="t")


def test_mock_handler_and_missing_tag():
    llm = MockLLM().on("echo", lambda msgs: msgs[-1]["content"])
    assert llm.chat([{"role": "user", "content": "hi"}], tag="echo") == "hi"
    with pytest.raises(LLMError):
        llm.chat([], tag="unknown")


def test_cassette_record_then_replay(tmp_path):
    msgs = [{"role": "user", "content": "q"}]
    recorder = CassetteLLM(tmp_path, inner=MockLLM(default=lambda m: "recorded"), model="m")
    assert recorder.chat(msgs, tag="plan") == "recorded"
    replayer = CassetteLLM(tmp_path, inner=None, model="m")
    assert replayer.chat(msgs, tag="plan") == "recorded"
    with pytest.raises(LLMError):
        replayer.chat([{"role": "user", "content": "other"}], tag="plan")


def test_live_mode_requires_key(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    settings = LLMSettings(base_url="http://x", model="m", api_key=None)
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        make_llm("live", settings=settings)


def test_secret_never_in_repr_or_errors():
    secret = "sk-test-SECRET-1234567890abcdef"
    settings = LLMSettings(base_url="http://x", model="m", api_key=secret)
    assert secret not in repr(settings)
    assert secret not in settings.model_dump_json()
    assert secret not in _redact(f"401 invalid key {secret}", settings)
