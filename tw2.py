from collections import deque

def best_first_search(start, goal, graph):
    visited = set()
    open_list = deque([[start, 0]])  # Use deque for efficient pop from the front
    closed_list = set()

    while open_list:
        # Pop the node with the lowest cost
        curNode, curCost = open_list.popleft()
        closed_list.add(curNode)

        print(f"At node {curNode}")
        print(f"CLOSED: {closed_list}")

        if curNode == goal:
            print("Goal node reached\n")
            return True

        for neighbor, cost in graph[curNode]:
            if neighbor not in visited and neighbor not in closed_list:
                open_list.append([neighbor, curCost + cost])
                visited.add(neighbor)

        print(f"UNSORTED OPEN: {list(open_list)}")
        open_list = deque(sorted(open_list, key=lambda x: x[1]))  # Sort by total path cost
        print(f"SORTED OPEN: {list(open_list)}\n")

    return False

# Example usage:
graph = {
    'A': [('B', 1), ('C', 3)],
    'B': [('D', 2), ('E', 4)],
    'C': [('F', 5)],
    'D': [],
    'E': [('G', 1)],
    'F': [('G', 2)],
    'G': []
}

start = 'A'
goal = 'G'
best_first_search(start, goal, graph)
