'''LLM 모델(OpenAI, Ollama 등) 초기화 및 관리 유틸리티'''
import os
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """LLM 클라이언트 기본 클래스"""
    
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, top_p: float = None) -> str:
        """텍스트 생성"""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API 클라이언트"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai 패키지가 설치되지 않았습니다: pip install openai")
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, top_p: float = None) -> str:
        """텍스트 생성"""
        try:
            params = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if top_p is not None:
                params["top_p"] = top_p

            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI API 오류: {str(e)}"


class OllamaClient(BaseLLMClient):
    """Ollama 로컬 LLM 클라이언트"""
    
    def __init__(self, base_url: str = "http://0.0.0.0:11434", model: str = "gemma3:4b"):
        self.base_url = base_url
        self.model = model
        
        try:
            import requests
            self.session = requests.Session()
        except ImportError:
            raise ImportError("requests 패키지가 설치되지 않았습니다: pip install requests")
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, top_p: float = None) -> str:
        """텍스트 생성"""
        try:
            options = {
                "temperature": temperature,
                "num_predict": max_tokens
            }
            if top_p is not None:
                options["top_p"] = top_p

            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": options
                }
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"Ollama API 오류: {response.status_code} {response.text}"
        except Exception as e:
            return f"Ollama 연결 오류: {str(e)}"


class MockLLMClient(BaseLLMClient):
    """테스트용 Mock LLM 클라이언트"""
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, top_p: float = None) -> str:
        """시뮬레이션 응답 생성"""
        return "Mock LLM 응답입니다. 1. search['mock action'] 2. choose['mock choice']"


class LLMManager:
    """LLM 클라이언트 관리자"""
    
    def __init__(self):
        self.client: Optional[BaseLLMClient] = None
        self._initialize_client()
    
    def _initialize_client(self):
        """환경변수 기반 클라이언트 초기화"""
        provider = os.getenv('LLM_PROVIDER', 'ollama').lower()
        
        if provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
            model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
            if not api_key:
                raise ValueError("LLM_PROVIDER가 'openai'일 경우 OPENAI_API_KEY가 필요합니다.")
            self.client = OpenAIClient(api_key, model)
        
        elif provider == 'ollama':
            base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            model = os.getenv('OLLAMA_MODEL', 'gemma3:4b')
            self.client = OllamaClient(base_url, model)
        
        else:
            print(f"알 수 없는 LLM_PROVIDER '{provider}', Mock 클라이언트를 사용합니다.")
            self.client = MockLLMClient()
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, top_p: float = None) -> str:
        """텍스트 생성"""
        if not self.client:
            raise ValueError("LLM 클라이언트가 초기화되지 않았습니다.")
        return self.client.generate(prompt, temperature, max_tokens, top_p)
    
    def get_client_info(self) -> Dict[str, Any]:
        """클라이언트 정보 반환"""
        if isinstance(self.client, OpenAIClient):
            return {"provider": "openai", "model": self.client.model}
        elif isinstance(self.client, OllamaClient):
            return {"provider": "ollama", "model": self.client.model, "base_url": self.client.base_url}
        else:
            return {"provider": "mock", "model": "mock"}


# 전역 LLM 매니저 인스턴스
_llm_manager = None


def get_llm_manager() -> LLMManager:
    """전역 LLM 매니저 인스턴스 반환"""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager


def generate_text(prompt: str, temperature: float = 0.7, max_tokens: int = 1000, top_p: float = None) -> str:
    """편의 함수: 텍스트 생성"""
    manager = get_llm_manager()
    return manager.generate(prompt, temperature, max_tokens, top_p)