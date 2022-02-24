"""
Data structures and utility functions needed by `prepare_data` and other modules. Consists of data 
structures for representing ConceptNet, normalization functions and a helper to load the ConceptNet 
pickle.
"""

from dataclasses import dataclass
from functools import reduce
import operator
from typing import Dict, Iterable, List, NamedTuple, Set, Tuple

import nltk
from nltk.stem import WordNetLemmatizer
import joblib

nltk.download("wordnet")
nltk.download('omw-1.4')

__lemmatizer = WordNetLemmatizer()


class EdgeDescriptor(NamedTuple):
    """Describes an edge in ConceptNet.

    Attributes
    ----------
    label_idx:
        index of the edge label in labels_idx2name
    weight:
        weight of the edge
    row_idx:
        index of the edge in "en_edges.csv" (for further information lookup if necessary)
    """

    label_idx : int
    weight : float
    row_idx : int


@dataclass
class ConceptNet:
    """This class contains the filtered and processed representation of the ConceptNet knowledge 
    graph.

    Attributes
    ----------

    nodes_idx2name:
        mapping node indices to normalized node names
    nodes_name2idx:
        mapping node names to indices

    labels_idx2name:
        mapping edge label indices to labels
    labels_name2idx:
        mapping edge labels to indices

    adjacency_lists:
        contains a list of neighbors for each node (indices are used). graph is represented as 
        undirected.
    edge_descriptors:
        maps a pair of indices to all direct edges in ConceptNet between this two nodes. edges are described via EdgeDescriptors. Edges are treated as *directed* here, i.e. if an edge is in `adjacency_lists`, but not in `edge_descriptors` it is an edge not originally present in ConceptNet and one must look up the reverse edge in `edge_descriptor`.
    """

    nodes_idx2name : List[str]
    nodes_name2idx : Dict[str, int]

    labels_idx2name : List[str]
    labels_name2idx : Dict[str, int]

    adjacency_lists : Dict[int, Set[int]]
    edge_descriptors : Dict[Tuple[int, int], Set[EdgeDescriptor]]

def removeprefix(s: str, prefix: str) -> str:
    if s.startswith(prefix):
        return s[len(prefix):]
    else:
        return s

def normalize_conceptnet(s: str) -> str:
    """Normalize a ConceptNet node to ensure that matching between example words and nodes in 
    ConceptNet works.

    Parameters
    ----------
    s : str
        string to normalize

    Returns
    -------
    str
        normalized string
    """

    s = removeprefix(s, "/c/en/")
    s = s.split("/")[0] # remove the optionally added (/n, /v, ...)
    s = s.replace("_", " ")
    s = s.casefold()
    s = __lemmatizer.lemmatize(s)

    return s

def normalize_input(s: str) -> str:
    """Normalize a input token to ensure that matching between example words and nodes in 
    ConceptNet works.

    Parameters
    ----------
    s : str
        string to normalize

    Returns
    -------
    str
        normalized string
    """
    # TODO switch to Spacy lemmatization

    s = s.casefold()
    s = __lemmatizer.lemmatize(s)

    return s

def prod(vals: Iterable[float]) -> float:
    return reduce(operator.mul, vals, 1)

def load_conceptnet(load_compressed: bool =False) -> ConceptNet:

    if load_compressed:
        return joblib.load("../data/processed/graph_representation_compressed.joblib")
    else:
        return joblib.load("../data/processed/graph_representation.joblib")
