from dotenv import load_dotenv

from graph.chains.hallucination_grader import GradeHallucinations, hallucination_grader

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

def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score


def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "In order to make pizza we need to first start with the dough",
        }
    )
    assert not res.binary_score
