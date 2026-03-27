import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-5-nano"), temperature=0)


class RouteQuery(BaseModel):
    """Route a user question to the appropriate data source."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        description="Route the question to 'vectorstore' or 'web_search'."
    )


structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are a router for a retrieval-augmented generation system.
Use 'vectorstore' for questions about agent memory, prompt engineering, or adversarial attacks on LLMs.
Use 'web_search' for questions that need broader or current web knowledge."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router
