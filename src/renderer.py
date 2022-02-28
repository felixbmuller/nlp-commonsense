from typing import List, Tuple
from utils import ConceptNet, removeprefix


def render_path_verbose(path: List[int], graph: ConceptNet):
    """this function gives a verbose textual representation for the given path, including all edges between the given nodes, node and label indices and label weights.

    Example
    -------
    > find_word_path("airport", "baggage", conceptnet, max_path_len=3, renderer=render_path_verbose)
    ['airport (35496)',
     '/r/AtLocation (idx 1, weight 3.464, reversed),/r/AtLocation (idx 1, weight 2.828, reversed)',
     'baggage (121612)']
    """

    if not path:
        return []

    rendered = [f"{graph.nodes_idx2name[path[0]]} ({path[0]})"]

    for path_idx, node_idx in enumerate(path[1:], start=1):
        prev_idx = path[
            path_idx - 1
        ]  # index of the previous node in path for edge lookup

        if (prev_idx, node_idx) in graph.edge_descriptors:
            edges = graph.edge_descriptors[(prev_idx, node_idx)]
            reverse_edge = False
        elif (node_idx, prev_idx) in graph.edge_descriptors:
            edges = graph.edge_descriptors[(node_idx, prev_idx)]
            reverse_edge = True
        else:
            raise ValueError(
                f"Illegal State: edge descriptors missing for edge present in graph ({node_idx}, {prev_idx})"
            )

        str_edge = ",".join(
            f"{graph.labels_idx2name[e.label_idx]} (idx {e.label_idx}, weight {e.weight}{', reversed' if reverse_edge else ''})"
            for e in edges
        )

        rendered.append(str_edge)
        rendered.append(f"{graph.nodes_idx2name[node_idx]} ({node_idx})")

    return rendered


def render_path_brief(path: List[int], graph: ConceptNet):
    """this function gives a brief textual representation for the given path.

    Example
    -------
    > find_word_path("airport", "baggage", conceptnet, max_path_len=3, renderer=render_path_verbose)
    airport <--AtLocation-- baggage
    """

    if not path:
        return []

    rendered = [graph.nodes_idx2name[path[0]]]

    for path_idx, node_idx in enumerate(path[1:], start=1):
        prev_idx = path[
            path_idx - 1
        ]  # index of the previous node in path for edge lookup

        if (prev_idx, node_idx) in graph.edge_descriptors:
            edges = graph.edge_descriptors[(prev_idx, node_idx)]
            reverse_edge = False
        elif (node_idx, prev_idx) in graph.edge_descriptors:
            edges = graph.edge_descriptors[(node_idx, prev_idx)]
            reverse_edge = True
        else:
            raise ValueError(
                f"Illegal State: edge descriptors missing for edge present in graph ({node_idx}, {prev_idx})"
            )

        best_edge = max(edges, key=lambda x: x.weight)
        best_edge_str = removeprefix(graph.labels_idx2name[best_edge.label_idx], "/r/")

        if not reverse_edge:
            best_edge_str = f"--{best_edge_str}-->"
        else:
            best_edge_str = f"<--{best_edge_str}--"

        rendered.append(best_edge_str)
        rendered.append(graph.nodes_idx2name[node_idx])

    return " ".join(rendered)


# from https://github.com/JoshFeldman95/Extracting-CK-from-Large-LM/blob/master/templates/relation_map.json
RELATION_MAP = {
    "RelatedTo": "{0} is like {1}",
    "ExternalURL": "{0} is described at the following URL {1}",
    "FormOf": "{0} is a form of the word {1}",
    "IsA": "{0} is {1}",
    "PartOf": "{1} has {0}",
    "HasA": "{0} has {1}",
    "UsedFor": "{0} is used for {1}",
    "CapableOf": "{0} can {1}",
    "AtLocation": "You are likely to find {0} in {1}",
    "Causes": "Sometimes {0} causes {1}",
    "HasSubevent": "Something you might do while {0} is {1}",
    "HasFirstSubevent": "the first thing you do when you {0} is {1}",
    "HasLastSubevent": "the last thing you do when you {0} is {1}",
    "HasPrerequisite": "something you need to do before you {0} is {1}",
    "HasProperty": "{0} is {1}",
    "MotivatedByGoal": "You would {0} because you want to {1}",
    "ObstructedBy": "{0} can be prevented by {1}",
    "Desires": "{0} wants {1}",
    "CreatedBy": "{1} is created by {0}",
    "Synonyms": "{0} and {1} have similar meanings",
    "Synonym": "{0} and {1} have similar meanings",
    "Antonym": "{0} is the opposite of {1}",
    "DistinctFrom": "it cannot be both {0} and {1}",
    "DerivedFrom": "the word {0} is derived from the word {1}",
    "SymbolOf": "{0} is a symbol of {1}",
    "DefinedAs": "{0} is defined as {1}",
    "Entails": "if {0} is happening, {1} is also happening",
    "MannerOf": "{0} is a specific way of doing {1}",
    "LocatedNear": "{0} is located near {1}",
    "dbpedia": "{0} is conceptually related to {1}",
    "SimlarTo": "{0} is similar to {1}",
    "EtymologicallyRelatedTo": "the word {0} and the word {1} have the same origin",
    "EtymologicallyDerivedFrom": "the word {0} comes from the word {1}",
    "CausesDesire": "{0} makes people want {1}",
    "MadeOf": "{0} is made of {1}",
    "ReceivesAction": "{0} can be {1} ",
    "InstanceOf": "{0} is an example of {1}",
    "NotDesires": "{0} does not want {1}",
    "NotUsedFor": "{0} is not used for {1}",
    "NotCapableOf": "{0} is not capable of {1}",
    "NotHasProperty": "{0} does not have the property of {1}",
    "NotMadeOf": "{0} is not made of {1}",
    "NotIsA": "{0} is not {1}",
    # New entries added based on ConceptNet
    "HasContext": "{0} is in the context of {1}",
    "SimilarTo": "{0} is similar to {1}",
    'dbpedia/capital': "{1} is the capital of {0}", # sic
    'dbpedia/field': "{0} is in the field of {1}",
    'dbpedia/genre': "the works of {0} are mainly {1}",
    'dbpedia/genus': "{1} is the genus of {0}",
    'dbpedia/influencedBy': "{0} was influenced by {1}",
    'dbpedia/knownFor': "{0} is known for {1}",
    'dbpedia/language': "{1} is the language of {0}",
    'dbpedia/leader': "{1} is the leader of {0}",
    'dbpedia/occupation': "{0}'s occupation is {1}",
    'dbpedia/product': "{0} produces {1}",
}


def render_path_natural(path: List[int], graph: ConceptNet) -> Tuple[str, List[float]]:
    """This function gives a natural language representation of the path using RELATION_MAP. It also returns the weight of all edges involved.

    Assignment 2.

    Example
    -------
    > find_word_path("airport", "baggage", conceptnet, max_path_len=3, renderer=render_path_verbose)
    airport <--AtLocation-- baggage
    """

    if not path:
        return "", []

    rendered = []
    weights = []

    for path_idx, node_idx in enumerate(path[1:], start=1):
        prev_idx = path[
            path_idx - 1
        ]  # index of the previous node in path for edge lookup

        if (prev_idx, node_idx) in graph.edge_descriptors:
            edges = graph.edge_descriptors[(prev_idx, node_idx)]
            reverse_edge = False
        elif (node_idx, prev_idx) in graph.edge_descriptors:
            edges = graph.edge_descriptors[(node_idx, prev_idx)]
            reverse_edge = True
        else:
            raise ValueError(
                f"Illegal State: edge descriptors missing for edge present in graph ({node_idx}, {prev_idx})"
            )

        best_edge = max(edges, key=lambda x: x.weight)
        best_edge_str = removeprefix(graph.labels_idx2name[best_edge.label_idx], "/r/")

        if best_edge_str not in RELATION_MAP:
            raise ValueError(f"edge type {best_edge_str} not in RELATION_MAP")

        if not reverse_edge:
            sentence = RELATION_MAP[best_edge_str].format(
                graph.nodes_idx2name[prev_idx], graph.nodes_idx2name[node_idx]
            )
        else:
            sentence = RELATION_MAP[best_edge_str].format(
                graph.nodes_idx2name[node_idx], graph.nodes_idx2name[prev_idx]
            )

        sentence = sentence + "."

        rendered.append(sentence)
        weights.append(best_edge.weight)

    return " ".join(rendered), weights
