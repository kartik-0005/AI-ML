import heapq

def astar(graph, start, goal):
    open_list = [(0, start)]
    parents = {}
    g_values = {node: float('inf') for node in graph}
    g_values[start] = 0
    f_values = {node: float('inf') for node in graph}
    f_values[start] = graph[start][1]

    iteration = 0

    while open_list:
        current_f, current_node = heapq.heappop(open_list)

        if current_node == goal:
            path = []
            while current_node in parents:
                path.append(current_node)
                current_node = parents[current_node]
            path.append(start)
            final_cost = g_values[goal]
            print(f"\nFinal Cost: {final_cost}")
            return path[::-1]

        for child, cost in graph[current_node][0].items():
            tentative_g = g_values[current_node] + cost
            if tentative_g < g_values[child]:
                parents[child] = current_node
                g_values[child] = tentative_g
                f_values[child] = tentative_g + graph[child][1]
                heapq.heappush(open_list, (f_values[child], child))

        iteration += 1
        print(f"\nIteration {iteration}:")
        print("Current Path:", reconstruct_path(parents, start, current_node))
        print(f"Evaluation Function Value for {current_node}: {f_values[current_node]}")

    return None

def reconstruct_path(parents, start, goal):
    path = [goal]
    while goal != start:
        goal = parents[goal]
        path.append(goal)
    return path[::-1]

# Example usage:
start_node = 'A'
goal_node = 'G'
graph = {
    'A': [{'B': 5, 'C': 10}, 10],
    'B': [{'D': 5, 'E': 5}, 7],
    'C': [{'F': 5}, 7],
    'D': [{'G': 10}, 3],
    'E': [{'G': 7}, 2],
    'F': [{'G': 8}, 1],
    'G': [{}, 0]
}

print("\nA* Search Path:")
path = astar(graph, start_node, goal_node)
print("Final Path:", path)
