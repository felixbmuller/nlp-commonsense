"""
This module includes the path finding algorithm and the textual visualization of paths. Entry point is `find_word_path`.
"""

from typing import Any, Dict, Iterable, List, Union
import logging

from utils import ConceptNet, normalize_input
from renderer import render_path_brief


def search_shortest_path(
    start_idx: int,
    end_idx: int,
    adjacency_lists: Dict[int, Iterable[int]],
    max_path_len,
) -> list:
    """the actual implementation of a BFS. This function is completely agnostic about ConceptNet, 
    it just works with adjacency lists of integers."""

    queue = [
        (start_idx, 0)
    ]  # nodes to be processed (tuple of node and path length to start node)
    predecessor_idx = {
        start_idx: -1
    }  # visited nodes, mapping each node to the idx of its predecessor

    while queue:
        node, path_len = queue.pop(0)

        #logging.debug(f"Processing {node} (path len {path_len})")

        if node == end_idx:

            #logging.debug("  Final node, building path")

            # build path in reverse
            path = [node]

            pred = predecessor_idx[node]
            while pred != -1:
                #logging.debug(f"    {pred}")
                path.insert(0, pred)
                pred = predecessor_idx[pred]

            return path

        for neighbour in adjacency_lists[node]:
            if neighbour in predecessor_idx:
                continue

            #logging.debug(f"  Processing unseen neighbor {neighbour}")

            predecessor_idx[neighbour] = node

            if path_len + 1 < max_path_len:
                #logging.debug("   Adding node to queue")
                queue.append((neighbour, path_len + 1))

    return []


def find_word_path(
        start_term: str, end_term: str, 
        graph: ConceptNet, 
        max_path_len: int =3,
        renderer=render_path_brief) -> Union[str, List[int]]:
    """Find the shortest path between `start_term` and `end_term` and return its textual 
    representation. 

    Parameters
    ----------
    start_term : str
        start term for path search
    end_term : str
        end term for path search
    graph : ConceptNet
        ConceptNet instance to work with
    max_path_len : int, optional
        maximal number of nodes in a path, by default 3
    renderer, optional
        function to visualize paths, by default render_path_brief. If None, the raw path (a list of int's) is returned.

    Returns
    -------
    str
        path visualization
    list[int]
        raw path, only returned if renderer is None
    """

    start_term = normalize_input(start_term)
    end_term = normalize_input(end_term)

    #logging.info(f"after normalization: {start_term}, {end_term}")

    if start_term in graph.nodes_name2idx:
        start_idx = graph.nodes_name2idx[start_term]
    else:
        #logging.warning(f"start {start_term} not in graph, skipping")
        return []

    if end_term in graph.nodes_name2idx:
        end_idx = graph.nodes_name2idx[end_term]
    else:
        #logging.warning(f"end {end_term} not in graph, skipping")
        return []

    path = search_shortest_path(
        start_idx, end_idx, graph.adjacency_lists, max_path_len=max_path_len
    )

    if renderer:
        return renderer(path, graph)
    else:
        return path


        