import os
from typing import Optional

from langchain_ollama import ChatOllama  # 로컬 LLM (Ollama)
from langchain_openai import ChatOpenAI  # 선택 의존성


class _DummyLLM:
    """Dummy LLM for testing purposes.
    When used, nodes.py detects this and replays expected actions from the offline log instead of calling a real LLM.
    """
    def __init__(self, temperature: float = 0.0):
        self.temperature = temperature
        self.is_dummy = True

def get_default_llm(model: Optional[str] = None, temperature: float = 0.0):
    """기본 LLM을 반환합니다.

    - OpenAI: 기본(provider=openai). OPENAI_API_KEY 필요.
    - Ollama: (provider=ollama) 또는 모델명이 "ollama:<model>" 형식이면 ChatOllama 사용.
    """

    # 1) 모델 인자에 "ollama:" 접두사를 허용
    if model and isinstance(model, str) and model.startswith("ollama:"):
        if ChatOllama is None:
            raise RuntimeError(
                "langchain-ollama가 설치되어 있지 않습니다. `pip install langchain-ollama`를 실행하세요."
            )
        ollama_model = model.split(":", 1)[1] or "gemma3:4b"
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        _np = os.getenv("OLLAMA_NUM_PREDICT")
        _nc = os.getenv("OLLAMA_NUM_CTX")
        _kwargs = {
            "model": ollama_model,
            "temperature": temperature,
            "base_url": base_url,
        }
        if _np:
            try:
                _kwargs["num_predict"] = int(_np)
            except Exception:
                pass
        if _nc:
            try:
                _kwargs["num_ctx"] = int(_nc)
            except Exception:
                pass
        return ChatOllama(**_kwargs)

    # 2) 환경변수 기반 provider 선택
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "dummy":
        return _DummyLLM(temperature=temperature)

    if provider == "ollama" or os.getenv("OLLAMA_MODEL"):
        if ChatOllama is None:
            raise RuntimeError(
                "langchain-ollama가 설치되어 있지 않습니다. `pip install langchain-ollama`를 실행하세요."
            )
        # Honor explicit model arg when provider is ollama
        ollama_model = (model or os.getenv("OLLAMA_MODEL") or "gemma3:4b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        _np = os.getenv("OLLAMA_NUM_PREDICT")
        _nc = os.getenv("OLLAMA_NUM_CTX")
        _kwargs = {
            "model": ollama_model,
            "temperature": temperature,
            "base_url": base_url,
        }
        if _np:
            try:
                _kwargs["num_predict"] = int(_np)
            except Exception:
                pass
        if _nc:
            try:
                _kwargs["num_ctx"] = int(_nc)
            except Exception:
                pass
        return ChatOllama(**_kwargs)

    # 3) 기본은 OpenAI
    if ChatOpenAI is None:
        raise RuntimeError(
            "langchain_openai가 설치되어 있지 않습니다. `pip install langchain-openai` 또는 LLM_PROVIDER=ollama와 함께 langchain-ollama를 사용하세요."
        )
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY가 설정되지 않았습니다. .env에 설정하거나 LLM_PROVIDER=ollama를 사용하세요."
        )
    return ChatOpenAI(model=model_name, temperature=temperature)
