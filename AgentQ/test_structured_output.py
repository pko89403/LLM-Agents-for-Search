from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# 1) 스키마 정의 (Pydantic v2)
class Book(BaseModel):
    title: str = Field(..., description="Book title")
    author: str = Field(..., description="Author full name")
    year: int = Field(..., description="Publication year")

# 2) Ollama LLM (로컬)
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
    # base_url="http://localhost:11434",  # 기본값이면 생략 가능
)

# 3) 구조화 출력 전용 LLM: 스키마를 직접 바인딩
#    -> LangChain이 JSON mode / tool-calling을 내부에서 사용해 스키마에 맞춰 반환
structured_llm = llm.with_structured_output(Book)  # ✅ 최신 방식

# 4) 프롬프트 (간결: 모델에게 "무엇을 뽑을지"만 전달)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Extract the following book info."),
        ("human", "Input: {text}"),
    ]
)

# 5) LangGraph 상태
class GraphState(TypedDict):
    text: str
    book: Optional[Book]
    error: Optional[str]

# 6) 노드: LLM 호출 → Pydantic 객체로 바로 수신
def node_extract(state: GraphState) -> GraphState:
    try:
        chain = prompt | structured_llm
        book: Book = chain.invoke({"text": state["text"]})
        return {**state, "book": book, "error": None}
    except Exception as e:
        return {**state, "book": None, "error": str(e)}

# 7) 그래프 구성/실행
graph = StateGraph(GraphState)
graph.add_node("extract_book", node_extract)
graph.add_edge(START, "extract_book")
graph.add_edge("extract_book", END)
app = graph.compile()

if __name__ == "__main__":
    out = app.invoke({"text": "The Hobbit by J.R.R. Tolkien (1937)"})
    print("Error:", out.get("error"))
    print("Book:", out.get("book"))