from platform import node
from typing import Iterable
import logging

from utils import ConceptNet, normalize_input


def search_shortest_path(
    start_idx: int,
    end_idx: int,
    adjacency_lists: dict[int, Iterable[int]],
    max_path_len=4,
) -> list:

    queue = [
        (start_idx, 0)
    ]  # nodes to be processed (tuple of node and path length to start node)
    predecessor_idx = {
        start_idx: -1
    }  # visited nodes, mapping each node to the idx of its predecessor

    while queue:
        node, path_len = queue.pop(0)

        logging.debug(f"Processing {node} (path len {path_len})")

        if node == end_idx:

            logging.debug("  Final node, building path")

            # build path in reverse
            path = [node]

            pred = predecessor_idx[node]
            while pred != -1:
                logging.debug(f"    {pred=}")
                path.insert(0, pred)
                pred = predecessor_idx[pred]

            return path

        for neighbour in adjacency_lists[node]:
            if neighbour in predecessor_idx:
                continue

            logging.debug(f"  Processing unseen neighbor {neighbour}")

            predecessor_idx[neighbour] = node

            if path_len + 1 < max_path_len:
                logging.debug("   Adding node to queue")
                queue.append((neighbour, path_len + 1))

    return []

def render_path_verbose(path: list[int], graph: ConceptNet):
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
        prev_idx = path[path_idx-1] # index of the previous node in path for edge lookup

        if (prev_idx, node_idx) in graph.edge_descriptors:
            edges = graph.edge_descriptors[(prev_idx, node_idx)]
            reverse_edge = False
        elif (node_idx, prev_idx) in graph.edge_descriptors:
            edges = graph.edge_descriptors[(node_idx, prev_idx)]
            reverse_edge = True
        else:
            raise ValueError(f"Illegal State: edge descriptors missing for edge present in graph ({node_idx=}, {prev_idx=})")

        str_edge = ",".join(f"{graph.labels_idx2name[e.label_idx]} (idx {e.label_idx}, weight {e.weight}{', reversed' if reverse_edge else ''})" for e in edges)

        rendered.append(str_edge)
        rendered.append(f"{graph.nodes_idx2name[node_idx]} ({node_idx})")

    return rendered

def render_path_brief(path: list[int], graph: ConceptNet):
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

    rendered = [graph.nodes_idx2name[path[0]]]

    for path_idx, node_idx in enumerate(path[1:], start=1):
        prev_idx = path[path_idx-1] # index of the previous node in path for edge lookup

        if (prev_idx, node_idx) in graph.edge_descriptors:
            edges = graph.edge_descriptors[(prev_idx, node_idx)]
            reverse_edge = False
        elif (node_idx, prev_idx) in graph.edge_descriptors:
            edges = graph.edge_descriptors[(node_idx, prev_idx)]
            reverse_edge = True
        else:
            raise ValueError(f"Illegal State: edge descriptors missing for edge present in graph ({node_idx=}, {prev_idx=})")

        best_edge = max(edges, key=lambda x: x.weight)
        best_edge_str = graph.labels_idx2name[best_edge.label_idx].removeprefix("/r/")

        if not reverse_edge:
            best_edge_str = f"--{best_edge_str}-->"
        else:
            best_edge_str = f"<--{best_edge_str}--"
        

        rendered.append(best_edge_str)
        rendered.append(graph.nodes_idx2name[node_idx])

    return " ".join(rendered)

def find_word_path(
        start_term: str, end_term: str, 
        graph: ConceptNet, 
        max_path_len=3,
        renderer=render_path_brief) -> list[str]:

    start_term = normalize_input(start_term)
    end_term = normalize_input(end_term)

    logging.info(f"after normalization: {start_term=}, {end_term=}")

    if start_term in graph.nodes_name2idx:
        start_idx = graph.nodes_name2idx[start_term]
    else:
        logging.warning(f"{start_term=} not in graph, skipping")
        return []

    if end_term in graph.nodes_name2idx:
        end_idx = graph.nodes_name2idx[end_term]
    else:
        logging.warning(f"{end_term=} not in graph, skipping")
        return []

    path = search_shortest_path(
        start_idx, end_idx, graph.adjacency_lists, max_path_len=max_path_len
    )

    return renderer(path, graph)



        