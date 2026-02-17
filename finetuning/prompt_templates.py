"""Prompt templates for metacognitive fine-tuning (Steyvers et al. method)."""

SYSTEM_PROMPT_CONFIDENCE = (
    "When answering a question, provide the answer and a confidence score "
    "between 0 and 1 for the answer."
)

ANSWER_FORMAT = "The answer is {answer} and my confidence score is {confidence:.2f}"

COMPARISON_PROMPT_TEMPLATE = (
    "Determine for which of the two following questions, your confidence "
    "score is higher\n\n"
    "Q1. {question1}\n\n"
    "Q2. {question2}\n\n"
    "Is your confidence in answering correctly higher for question Q1 or "
    "question Q2?"
)

COMPARISON_ANSWER_FORMAT = "The answer is Q{choice}"


def format_single_question_messages(question, answer=None, confidence=None):
    """
    Format a single question as Gemma chat messages.

    If answer and confidence are provided, includes the assistant response
    (for training). Otherwise, only the user turn (for inference).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_CONFIDENCE},
        {"role": "user", "content": question},
    ]
    if answer is not None and confidence is not None:
        messages.append({
            "role": "assistant",
            "content": ANSWER_FORMAT.format(answer=answer, confidence=confidence),
        })
    return messages


def format_comparison_messages(question1, question2, choice=None):
    """
    Format a pairwise comparison as Gemma chat messages.

    choice: 1 or 2 (which question has higher confidence).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_CONFIDENCE},
        {
            "role": "user",
            "content": COMPARISON_PROMPT_TEMPLATE.format(
                question1=question1, question2=question2,
            ),
        },
    ]
    if choice is not None:
        messages.append({
            "role": "assistant",
            "content": COMPARISON_ANSWER_FORMAT.format(choice=choice),
        })
    return messages
