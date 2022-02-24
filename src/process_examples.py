"""
Functions to transform examples.txt into machine readable form and to extract terms from example 
questions.
"""

from typing import Dict, List, Set, Tuple, Union
import itertools

import spacy

import utils as U

nlp = spacy.load("en_core_web_sm")
stopwords = set(nlp.Defaults.stop_words).union({",", ".", "?", ":", ";"})


def load_examples() -> List[Dict[str, Union[str, List[str]]]]:
    """Quick and dirty parser to turn examples.txt into a machine-readable form.

    Returns
    -------
    list[dict[str, Union[str, list[str]]]]
        list of examples. Each examples is of the form
            {
                "question": QUESTION,
                "context": CONTEXT or "",
                "choices": [CHOICE1, CHOICE2, ...]
            }
    """

    with open("../data/raw/examples.txt") as fp:
        data = fp.read()

    examples = data.split("#")[1:]

    parsed = []

    for e in examples:
        parts = e.split("\n")

        question = next(
            (
                U.removeprefix(p, "Question:").strip()
                for p in parts
                if p.startswith("Question: ")
            ),
            "",
        )
        context = next(
            (
                U.removeprefix(p, "Context:").strip()
                for p in parts
                if p.startswith("Context: ")
            ),
            "",
        )
        choices = [p.split(")")[1].strip() for p in parts if p.startswith("(")]

        parsed.append({"question": question, "context": context, "choices": choices})

    return parsed


def extract_terms(input: str) -> Set[str]:
    """extract terms from a string. Terms are all tokens and noun chunks that are no stopwords. Spacy is used for processing

    Parameters
    ----------
    input : str
        string to extract terms from

    Returns
    -------
    set[str]
        set of terms
    """
    doc = nlp(input)

    tokens = [*doc, *doc.noun_chunks]

    token_texts = [
        U.normalize_input(t.text)
        for t in tokens
        if U.normalize_input(t.text) not in stopwords
    ]
    token_texts = set(
        U.removeprefix(U.removeprefix(U.removeprefix(t, "the "), "a "), "an ")
        for t in token_texts
    )

    return token_texts


def extract_terms_from_example(example: dict) -> Tuple[Set[str], Set[str]]:
    """extract terms from an example using `extract_terms`.

    Parameters
    ----------
    example : dict
        example as returned by `load_examples`.

    Returns
    -------
    set[str]
        all terms appearing in question and context
    set[str]]
        all terms appearing in one of the answer choices
    """

    question_context = set(
        itertools.chain(
            extract_terms(example["question"]), extract_terms(example["context"])
        )
    )

    choices = set(
        itertools.chain.from_iterable(extract_terms(c) for c in example["choices"])
    )

    return question_context, choices
