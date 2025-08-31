"""
LLM 모델 초기화 및 관리 유틸리티
OpenAI, Ollama 등 다양한 LLM 지원
"""

import os
from typing import Optional, Dict, Any, List
from typing import Type
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage


class LLMManager:
    """LLM 관리 클래스"""

    def __init__(self):
        self._models: Dict[str, BaseChatModel] = {}
        self._default_model: Optional[str] = None

    def add_openai_model(
        self,
        name: str = "openai",
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs
    ) -> BaseChatModel:
        """OpenAI 모델 추가"""
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API 키가 필요합니다. OPENAI_API_KEY 환경변수를 설정하거나 api_key 매개변수를 제공하세요.")

        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            **kwargs
        )

        self._models[name] = llm
        if self._default_model is None:
            self._default_model = name

        print(f"✅ OpenAI 모델 '{name}' 추가됨 (모델: {model})")
        return llm

    def add_ollama_model(
        self,
        name: str = "ollama",
        model: str = "qwen2.5:7b-instruct-q4_K_M",
        base_url: str = None,
        temperature: float = 0.0,
        **kwargs
    ) -> BaseChatModel:
        """Ollama 모델 추가"""
        # 환경변수에서 Ollama 호스트 확인
        import os
        if base_url is None:
            ollama_host = os.getenv("OLLAMA_HOST", "localhost:11434")
            if not ollama_host.startswith("http"):
                base_url = f"http://{ollama_host}"
            else:
                base_url = ollama_host

        llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            **kwargs
        )

        self._models[name] = llm
        if self._default_model is None:
            self._default_model = name

        print(f"✅ Ollama 모델 '{name}' 추가됨 (모델: {model})")
        return llm

    def get_model(self, name: Optional[str] = None) -> BaseChatModel:
        """모델 가져오기"""
        model_name = name or self._default_model
        if not model_name or model_name not in self._models:
            raise ValueError(f"모델 '{model_name}'을 찾을 수 없습니다. 사용 가능한 모델: {list(self._models.keys())}")

        return self._models[model_name]

    def list_models(self) -> List[str]:
        """등록된 모델 목록 반환"""
        return list(self._models.keys())

    async def invoke_model(
        self,
        messages: List[BaseMessage],
        model_name: Optional[str] = None,
        **kwargs
    ) -> AIMessage:
        """모델 호출"""
        model = self.get_model(model_name)
        return await model.ainvoke(messages, **kwargs)

    async def invoke_with_system(
        self,
        system_prompt: str,
        user_message: str,
        model_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """시스템 프롬프트와 사용자 메시지로 모델 호출"""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]

        response = await self.invoke_model(messages, model_name, **kwargs)
        return response.content

    def supports_structured_output(self, model_name: Optional[str] = None) -> bool:
        model = self.get_model(model_name)
        return hasattr(model, "with_structured_output")

    async def invoke_structured_with_system(
        self,
        system_prompt: str,
        user_message: str,
        schema_model: Type[BaseModel],
        model_name: Optional[str] = None,
        **kwargs
    ):
        """
        Try to call the underlying model with structured output (Pydantic schema).
        Returns the parsed object on success, or None if unsupported / failed.
        """
        model = self.get_model(model_name)
        try:
            if not hasattr(model, "with_structured_output"):
                return None
            structured_llm = model.with_structured_output(schema_model)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]
            resp = await structured_llm.ainvoke(messages, **kwargs)
            return resp
        except Exception as e:
            print(f"⚠️ Structured output invoke 실패: {e}")
            return None


# 전역 LLM 매니저 인스턴스
_llm_manager: Optional[LLMManager] = None


def get_llm_manager() -> LLMManager:
    """LLM 매니저 싱글톤 인스턴스 반환"""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager


def setup_default_llms():
    """기본 LLM 설정"""
    manager = get_llm_manager()

    # OpenAI 모델 설정 시도
    try:
        if os.getenv("OPENAI_API_KEY"):
            manager.add_openai_model()
            print("🤖 OpenAI 모델이 기본으로 설정되었습니다.")
        else:
            print("⚠️ OPENAI_API_KEY가 설정되지 않았습니다.")
    except Exception as e:
        print(f"⚠️ OpenAI 모델 설정 실패: {e}")

    # Ollama 모델 설정 시도
    try:
        manager.add_ollama_model()
        print("🦙 Ollama 모델이 추가되었습니다.")
    except Exception as e:
        print(f"⚠️ Ollama 모델 설정 실패: {e}")

    # 사용 가능한 모델 확인
    available_models = manager.list_models()
    if not available_models:
        print("❌ 사용 가능한 LLM이 없습니다.")
        print("💡 해결 방법:")
        print("   1. OPENAI_API_KEY 환경변수 설정")
        print("   2. Ollama 서버 실행 (ollama serve)")
    else:
        print(f"✅ 사용 가능한 모델: {available_models}")


async def test_llm_connection(model_name: Optional[str] = None) -> bool:
    """LLM 연결 테스트"""
    try:
        manager = get_llm_manager()
        response = await manager.invoke_with_system(
            system_prompt="You are a helpful assistant.",
            user_message="Hello! Please respond with 'Connection successful!'",
            model_name=model_name
        )

        if "successful" in response.lower():
            print(f"✅ LLM 연결 테스트 성공: {model_name or 'default'}")
            return True
        else:
            print(f"⚠️ LLM 응답이 예상과 다름: {response}")
            return False

    except Exception as e:
        print(f"❌ LLM 연결 테스트 실패: {e}")
        return False


# 편의 함수들
async def call_llm(
    system_prompt: str,
    user_message: str,
    model_name: Optional[str] = None
) -> str:
    """간단한 LLM 호출"""
    manager = get_llm_manager()
    return await manager.invoke_with_system(system_prompt, user_message, model_name)


async def call_llm_with_messages(
    messages: List[BaseMessage],
    model_name: Optional[str] = None
) -> str:
    """메시지 리스트로 LLM 호출"""
    manager = get_llm_manager()
    response = await manager.invoke_model(messages, model_name)
    return response.content
