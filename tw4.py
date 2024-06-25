def minmax(node, tree, is_max_node):
    if not tree[node][0]:  # If leaf node
        print(f"Leaf node {node} has value {tree[node][1]}")
        return tree[node][1]

    children = tree[node][0]

    if is_max_node:
        value = -float('inf')
        print(f"Max node {node} with children {children}")
        for child in children:
            value = max(value, minmax(child, tree, False))
    else:
        value = float('inf')
        print(f"Min node {node} with children {children}")
        for child in children:
            value = min(value, minmax(child, tree, True))

    tree[node][1] = value
    print(f"Node {node} computed value {value}")
    return value

# Provided tree structure
tree = {
    'A': [['B', 'C'], None],
    'B': [['D', 'E'], None],
    'C': [['F', 'G'], None],
    'D': [['H', 'I'], None],
    'E': [['J', 'K'], None],
    'F': [['L', 'M'], None],
    'G': [['N', 'O'], None],
    'H': [[], -1],
    'I': [[], 4],
    'J': [[], 3],
    'K': [[], 6],
    'L': [[], -3],
    'M': [[], -5],
    'N': [[], 0],
    'O': [[], 7]
}

# Starting the minimax algorithm from the root node 'A' assuming it's a max node
result = minmax('A', tree, True)
print(f"\nThe most appropriate value for the root node 'A' is {result}")
