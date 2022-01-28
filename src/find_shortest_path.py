
def search_shortest_path(
    start_idx: int, 
    end_idx: int, 
    adjacency_lists: dict[int, list[int]], 
    max_visits=10000) -> list:

    # TODO limit on path length here

    queue = [start_idx] # nodes to be processed
    predecessor_idx = {start_idx: -1} # visited nodes, mapping each node to the idx of its predecessor
    while queue and len(predecessor_idx) < max_visits:
        node = queue.pop(0)

        for neighbour in adjacency_lists[node]:
            if neighbour in predecessor_idx:
                continue

            predecessor_idx[neighbour] = node
            queue.append(neighbour)

            if neighbour == end_idx:
                
                path = [neighbour]

                pred = predecessor_idx[neighbour]
                while pred != -1:
                    path.insert(0, pred)

                return path

            

    return []