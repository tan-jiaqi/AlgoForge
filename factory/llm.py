"""LLM 访问层。

所有 Agent 都通过 :class:`LLM` 调用大模型，支持四种模式：

* ``mock``   —— :class:`MockLLM`，按调用标签（tag）返回确定结果，用于测试。
* ``live``   —— :class:`OpenAILLM`，对接任意 OpenAI 兼容接口，带磁盘缓存。
* ``record`` —— 真实调用，同时把响应录制为 cassette，供之后回放。
* ``replay`` —— 只回放已录制的 cassette，完全离线、可复现。

密钥处理：API Key 只从环境变量 ``LLM_API_KEY`` 读取（``.env`` 已被 git 忽略），
以 ``SecretStr`` 保存，不会出现在 repr 或日志里，也不参与缓存/cassette 的键和内容。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr, ValidationError

from factory.utils import CONFIG_DIR, ROOT, get_logger

log = get_logger(__name__)

Message = dict[str, Any]
T = TypeVar("T", bound=BaseModel)
Mode = Literal["mock", "live", "record", "replay"]

DEFAULT_CASSETTE_DIR = ROOT / "examples" / "cassettes"


class LLMError(RuntimeError):
    pass


class LLMSettings(BaseModel):
    base_url: str
    model: str
    api_key: SecretStr | None = None
    temperature: float = 0.2
    timeout: float = 60
    max_retries: int = 2
    cache_dir: str = ".cache/llm"

    @classmethod
    def load(cls, path: Path | None = None) -> "LLMSettings":
        """读取 ``configs/llm.yaml``，再用环境变量（含 ``.env``）覆盖。"""
        load_dotenv(ROOT / ".env")
        cfg = yaml.safe_load((path or CONFIG_DIR / "llm.yaml").read_text(encoding="utf-8"))
        if url := os.getenv("LLM_BASE_URL"):
            cfg["base_url"] = url
        if model := os.getenv("LLM_MODEL"):
            cfg["model"] = model
        key = os.getenv("LLM_API_KEY")
        cfg["api_key"] = SecretStr(key) if key else None
        return cls(**cfg)


# ---------------------------------------------------------------------------
# 基础接口
# ---------------------------------------------------------------------------

class LLM:
    """最小的对话接口，加上按 schema 校验的结构化输出。"""

    model: str = "unknown"

    def chat(self, messages: list[Message], *, tag: str = "", json_mode: bool = False) -> str:
        raise NotImplementedError

    def structured(
        self,
        messages: list[Message],
        schema: type[T],
        *,
        tag: str = "",
        check: Callable[[T], None] | None = None,
        attempts: int = 2,
    ) -> T:
        """要求返回符合 ``schema`` 的 JSON；校验失败时带上错误信息重问，最多 ``attempts`` 次。

        ``check`` 用于 schema 之外的语义校验（如算子是否存在），发现问题时抛出 ``ValueError``。
        """
        instruction = (
            "Respond with a single JSON object only, no prose, conforming to this JSON schema:\n"
            + json.dumps(schema.model_json_schema(), ensure_ascii=False)
        )
        convo = [*messages, {"role": "system", "content": instruction}]
        last_error = ""
        for attempt in range(attempts):
            raw = self.chat(convo, tag=tag, json_mode=True)
            try:
                obj = schema.model_validate(extract_json(raw))
                if check is not None:
                    check(obj)
                return obj
            except (ValueError, ValidationError) as exc:
                last_error = str(exc)
                log.warning("structured output for %s failed validation (attempt %d)", tag or schema.__name__, attempt + 1)
                convo = [
                    *convo,
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"Your JSON was invalid:\n{last_error}\nReturn corrected JSON only."},
                ]
        raise LLMError(f"could not obtain valid {schema.__name__}: {last_error}")


def extract_json(text: str) -> Any:
    """从 LLM 回复中解析 JSON，容忍代码块标记和前后的说明文字。"""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise ValueError("no JSON object found in LLM response") from None


def request_hash(model: str, messages: list[Message], json_mode: bool) -> str:
    """缓存/回放用的稳定键，刻意不包含密钥和接口地址。"""
    payload = json.dumps({"model": model, "messages": messages, "json": json_mode}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 具体实现
# ---------------------------------------------------------------------------

class OpenAILLM(LLM):
    """OpenAI 兼容 chat completions 接口的客户端。"""

    def __init__(self, settings: LLMSettings):
        if settings.api_key is None:
            raise LLMError("LLM_API_KEY is not set; copy .env.example to .env and fill it in")
        from openai import OpenAI

        self.settings = settings
        self.model = settings.model
        self._client = OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key.get_secret_value(),
            timeout=settings.timeout,
            max_retries=settings.max_retries,
        )

    def chat(self, messages: list[Message], *, tag: str = "", json_mode: bool = False) -> str:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": self.settings.temperature}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            try:
                resp = self._client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                if not (json_mode and "response_format" in str(exc)):
                    raise
                kwargs.pop("response_format")  # 部分网关不支持 JSON 模式；prompt 中已要求输出 JSON
                resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - 统一 SDK/网络错误，且绝不泄露密钥
            raise LLMError(f"LLM call failed ({type(exc).__name__}): {_redact(str(exc), self.settings)}") from None
        self.last_usage = resp.usage.model_dump() if resp.usage else None
        return resp.choices[0].message.content or ""

    def list_models(self) -> list[str]:
        try:
            return sorted(m.id for m in self._client.models.list())
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"listing models failed ({type(exc).__name__}): {_redact(str(exc), self.settings)}") from None


class CassetteLLM(LLM):
    """包装另一个 LLM 的磁盘响应存储。

    设置了 ``inner`` 时作为读穿缓存/录制器；``inner=None`` 时为严格回放器，
    遇到未录制的请求直接报错。
    """

    def __init__(self, directory: Path, inner: LLM | None, model: str, *, agent_namespace: bool = False):
        self.directory = Path(directory)
        self.inner = inner
        self.model = model
        self.calls: list[dict] = []
        self.agent_namespace = agent_namespace

    def chat(self, messages: list[Message], *, tag: str = "", json_mode: bool = False) -> str:
        key = request_hash(self.model, messages, json_mode)
        directory = self.directory / "agent-v2" if self.agent_namespace and tag in {"plan_spec", "plan", "synthesize", "repair", "lesson"} else self.directory
        path = directory / f"{tag or 'chat'}-{key}.json"
        start = time.perf_counter()
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            self.calls.append({"tag": tag, "request_hash": key, "model": self.model,
                               "source": "replay", "seconds": time.perf_counter() - start,
                               "recorded_seconds": record.get("seconds"), "usage": record.get("usage")})
            return record["response"]
        if self.inner is None:
            self.calls.append({"tag": tag, "request_hash": key, "model": self.model, "source": "missing_recording"})
            raise LLMError(f"no recorded response for {path.name}; run once with --llm record")
        try:
            response = self.inner.chat(messages, tag=tag, json_mode=json_mode)
        except LLMError:
            self.calls.append({"tag": tag, "request_hash": key, "model": self.model,
                               "source": "remote_error", "seconds": time.perf_counter() - start})
            raise
        seconds = time.perf_counter() - start
        usage = getattr(self.inner, "last_usage", None)
        origin = "remote" if isinstance(self.inner, OpenAILLM) else "provided"
        self.calls.append({"tag": tag, "request_hash": key, "model": self.model,
                           "source": origin, "seconds": seconds, "usage": usage})
        directory.mkdir(parents=True, exist_ok=True)
        record = {"tag": tag, "model": self.model, "messages": messages, "response": response,
                  "json_mode": json_mode, "seconds": seconds, "usage": usage, "source": origin}
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return response


class TrialLLM(LLM):
    """在每次请求前加一条试验编号，用于独立重复实验。

    部分 OpenAI 兼容网关对内容相同的请求返回缓存回复，即使 temperature 较高也一样，
    重复运行因此不是独立样本。编号在录制层之前加入，回放时按同一编号重建请求即可命中录制。
    """

    def __init__(self, inner: LLM, trial: str):
        self.inner = inner
        self.trial = trial
        self.model = inner.model

    @property
    def calls(self):
        return getattr(self.inner, "calls", [])

    @property
    def directory(self):
        return self.inner.directory

    def chat(self, messages: list[Message], *, tag: str = "", json_mode: bool = False) -> str:
        note = {"role": "system", "content": f"trial={self.trial}（独立重复实验编号，与任务无关）"}
        return self.inner.chat([note, *messages], tag=tag, json_mode=json_mode)


Handler = Callable[[list[Message]], str]


@dataclass
class MockCall:
    tag: str
    messages: list[Message]
    response: str


@dataclass
class MockLLM(LLM):
    """用于测试和离线演示的确定性 LLM。

    响应来源按优先级依次为：该 tag 排队的响应、为该 tag 注册的处理函数、``default``。
    """

    handlers: dict[str, Handler] = field(default_factory=dict)
    default: Handler | None = None
    model: str = "mock"
    calls: list[MockCall] = field(default_factory=list)
    _queues: dict[str, list[str]] = field(default_factory=dict)

    def on(self, tag: str, handler: Handler) -> "MockLLM":
        self.handlers[tag] = handler
        return self

    def queue(self, tag: str, *responses: str | BaseModel) -> "MockLLM":
        texts = [r.model_dump_json() if isinstance(r, BaseModel) else r for r in responses]
        self._queues.setdefault(tag, []).extend(texts)
        return self

    def chat(self, messages: list[Message], *, tag: str = "", json_mode: bool = False) -> str:
        if self._queues.get(tag):
            response = self._queues[tag].pop(0)
        elif tag in self.handlers:
            response = self.handlers[tag](messages)
        elif self.default is not None:
            response = self.default(messages)
        else:
            raise LLMError(f"MockLLM has no response for tag {tag!r}")
        self.calls.append(MockCall(tag, messages, response))
        return response


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def make_llm(mode: Mode = "live", *, settings: LLMSettings | None = None, cassette_dir: Path | None = None) -> LLM:
    if mode == "mock":
        from factory.agents.rules import rule_based_mock  # 延迟导入，避免循环依赖

        return rule_based_mock()
    settings = settings or LLMSettings.load()
    cassettes = cassette_dir or DEFAULT_CASSETTE_DIR
    if mode == "replay":
        return CassetteLLM(cassettes, inner=None, model=settings.model, agent_namespace=cassette_dir is None)
    live = OpenAILLM(settings)
    if mode == "record":
        return CassetteLLM(cassettes, inner=live, model=settings.model, agent_namespace=cassette_dir is None)
    return CassetteLLM(ROOT / settings.cache_dir, inner=live, model=settings.model)


def _redact(text: str, settings: LLMSettings) -> str:
    """在展示任何文本（如 SDK 报错）前，去掉其中的 API Key。"""
    if settings.api_key is not None:
        secret = settings.api_key.get_secret_value()
        if secret:
            text = text.replace(secret, "***")
    return re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "sk-***", text)
