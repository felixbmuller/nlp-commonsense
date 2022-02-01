"""
This script  transforms the raw ConceptNet data into a joblib-pickled instance of `utils.ConceptNet`
that can later be used for path extraction. It only needs to be executed once and should be executed
via command line. 

Steps
-----
1. Save the ConceptNet data under `../data/raw/conceptnet-assertions-5.7.0.csv.gz`
2. Run `python prepare_data.py filter-conceptnet-en` to generate `../data/processed/en_edges.csv`
3. Run `python prepare_data.py process-conceptnet-csv` to generate `../data/processed/graph_representation.joblib`
"""

from collections import defaultdict
import gzip
import json

import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
import fire

import utils as U

LINE_COUNT = 34074917  # number of lines in conceptnet-assertions-5.7.0.csv.gz


def filter_conceptnet_en():
    """Read the raw conceptnet dump (.csv.gz), filter all edges between /c/en/* nodes and save them
    to processed/en_edges.csv.
    """

    raw_edges = []

    with gzip.open(
        "../data/raw/conceptnet-assertions-5.7.0.csv.gz", mode="rt", encoding="utf-8"
    ) as fp:
        for line in tqdm(fp, total=LINE_COUNT):
            parts = line.split("\t")
            assert len(parts) == 5, f"Bad line: {parts}"
            if not (parts[2].startswith("/c/en/") and parts[3].startswith("/c/en/")):
                continue
            raw_edges.append(parts)

    df = pd.DataFrame(raw_edges, columns=["uri", "label", "start", "end", "info"])

    df.to_csv("../data/processed/en_edges.csv")


def process_conceptnet_csv():
    """Build a ConceptNet object out of en_edges.csv and store it as graph_representation.joblib.
    """

    df = pd.read_csv("../data/processed/en_edges.csv")

    edges = list(sorted(df.label.unique()))

    # perform lemmatization etc to ensure that matching between nodes and words in examples works
    nodes = [U.normalize_conceptnet(s) for s in tqdm(sorted(set(df.start).union(set(df.end))))]

    nodes2idx = {node: idx for idx, node in enumerate(nodes)}
    edges2idx = {edge: idx for idx, edge in enumerate(edges)}

    adjacency_list = defaultdict(set)
    edges2description = defaultdict(set)

    for row_idx, row in tqdm(df.iterrows(), total=len(df)):
        start_idx = nodes2idx[U.normalize_conceptnet(row["start"])]
        end_idx = nodes2idx[U.normalize_conceptnet(row["end"])]

        adjacency_list[start_idx].add(end_idx)
        adjacency_list[end_idx].add(start_idx)  # undirected graph

        weight = json.loads(row["info"]).get("weight", np.nan)

        edges2description[(start_idx, end_idx)].add(
            U.EdgeDescriptor(edges2idx[row["label"]], weight, row_idx)
        )

    graph_repr = U.ConceptNet(
        nodes, nodes2idx, edges, edges2idx, adjacency_list, edges2description
    )

    joblib.dump(graph_repr, "../data/processed/graph_representation.joblib")

if __name__ == "__main__":
    fire.Fire({
        "filter-conceptnet-en": filter_conceptnet_en,
        "process-conceptnet-csv": process_conceptnet_csv,
    })