from typing import List

from process_examples import extract_terms
from find_shortest_path import find_word_path
from renderer import render_path_natural
from utils import ConceptNet, prod


def get_knowledge_for_example(
    premise_question: str, choice: str, conceptnet: ConceptNet, max_paths: int, raw_output:bool=False
) -> str:
    """Return a list of paths connecting terms from the premise with terms from the choice. Paths are extracted from a knowledge base and encoded in natural language. If more than max_paths paths are found, paths are selected primarily based lower number of nodes and secondarily on higher product of edge weights.

    Assignment 2.

    Parameters
    ----------
    premise_question : str
        premise/question sentence from a Q&A example
    choice : str
        choice/hypothesis from a Q&A example
    conceptnet : ConceptNet
        knoqledge base
    max_paths : int
        maximum number of paths to return

    Returns
    -------
    str
        concatenated paths, each encoded as natural language
    """

    p_terms = extract_terms(premise_question)
    c_terms = extract_terms(choice)

    paths = []

    for tp in p_terms:
        for tc in c_terms:
            p = find_word_path(tp, tc, conceptnet, renderer=None)

            context, weights = render_path_natural(p, conceptnet)

            if context:
                paths.append((context, weights))

    if len(paths) > max_paths:
        # Too many paths, need to select the most relevant ones

        # Sort the paths using the ascending number of edges (= number of weights) as primary key
        # and the descending product of weights as secondary key
        paths.sort(key=lambda x: prod(x[1]), reverse=True)
        paths.sort(key=lambda x: len(x[1]))

        paths = paths[:max_paths]

    if not raw_output:
        return " ".join(context for context, _ in paths)
    else:
        return paths
