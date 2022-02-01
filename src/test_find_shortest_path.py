"""
some unit tests to ensure that the BFS works as expected.
"""

import unittest

from find_shortest_path import *

class FindShortestPathTest(unittest.TestCase):

    def test_simple(self):

        adj = {
            0: [1],
            1: [0, 2],
            2: [1]
        }

        path = search_shortest_path(0, 2, adj)
        path_rev = search_shortest_path(2, 0, adj)

        self.assertListEqual(path, [0, 1, 2])
        self.assertListEqual(path_rev, [2, 1, 0])

    def test_graph(self):

        adj = {
            0: [1, 3],
            1: [0, 2, 3],
            2: [1, 4],
            3: [0, 1, 4],
            4: [2, 3]
        }

        path = search_shortest_path(0, 2, adj)
        path_rev = search_shortest_path(2, 0, adj)

        self.assertListEqual(path, [0, 1, 2])
        self.assertListEqual(path_rev, [2, 1, 0])

    def test_path_too_long(self):

        adj = {
            0: [1],
            1: [0, 2],
            2: [1]
        }

        path = search_shortest_path(0, 2, adj, max_path_len=2)
        self.assertListEqual(path, [])

        adj2 = {
            0: [1],
            1: [0, 2],
            2: [1, 3],
            3: [2, 4],
            4: [3],
        }

        path = search_shortest_path(0, 4, adj2)
        self.assertListEqual(path, [])

    def test_no_path(self):

        adj = {
            0: [1],
            1: [0],
            2: [3],
            3: [2],
        }

        path = search_shortest_path(0, 2, adj)
        self.assertListEqual(path, [])

    def test_edge_cases(self):

        adj = {0: set()}

        self.assertListEqual(search_shortest_path(0, 0, adj), [0])

        adj = {0: set(), 1: set()}

        self.assertListEqual(search_shortest_path(0, 1, adj), [])


if __name__ == "__main__":
    unittest.main()