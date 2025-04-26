from pydantic import BaseModel, Field
from typing import List


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous.")


class AnswerQuestion(BaseModel):
    """Answer the Question."""

    answer: str = Field(description="~250 word detailed answer to the question.")
    search_queries: List[str] = Field(
        description="1-3 search queries seperately for researching improvements to address the critique of the current answer."
    )
    reflection: Reflection = Field(description="Your reflection on the initial answer.")


class ReviseAnswer(AnswerQuestion):
    """Revise the Answer."""

    references: List[str] = Field(
        description="Citations motivating your updated answers."
    )
