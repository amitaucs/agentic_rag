import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-5-nano"), temperature=0)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions using the provided context. "
            "If the answer is not in the context, say you do not know. "
            "Keep the answer concise.",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)

generation_chain = prompt | llm | StrOutputParser()
