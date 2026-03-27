from dotenv import load_dotenv
load_dotenv()

from graph.chains.retrieval_grader import retrieval_grader,GradeDocument
from ingestion import retriever
from graph.chains.generations import generation_chain
from graph.chains.router import RouteQuery, question_router

def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_text = docs[1].page_content

    res: GradeDocument = retrieval_grader.invoke(
        {"question": question, "document": doc_text}
    )

    assert res.binary_score == "yes"


def test_retrieval_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_text = docs[0].page_content

    res: GradeDocument = retrieval_grader.invoke(
        {"question": "How to make Pizza", "document": doc_text}
    )

    assert res.binary_score == 'no'

def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    print(generation)

