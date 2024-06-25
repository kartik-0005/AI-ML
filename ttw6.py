import numpy as np

def unit_step(x, t):
    return 1 if x > t else 0

def perceptron_learn(inputs, outputs, weights, learning_rate, threshold, epochs=30):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        is_updated = False
        for i, x in enumerate(inputs):
            weighted_sum = np.dot(weights, x)
            prediction = unit_step(weighted_sum, threshold)
            error = outputs[i] - prediction
            print(f"Instance {i + 1}: {x}, Target: {outputs[i]}, Predicted: {prediction}")
            if error != 0:
                is_updated = True
                weights += learning_rate * x * error
                print(f"Updated weights: {weights}")
            else:
                print("Output matches. No update needed.")
        if not is_updated:
            print("\nFinal weights:", weights)
            break
        print("")
    return weights

# Example usage:
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs_or = np.array([0, 1, 1, 1])
outputs_and = np.array([0, 0, 0, 1])
inputs_not = np.array([[0], [1]])
outputs_not = np.array([1, 0])

print("-----------------------------------------------")
print("OR gate")
perceptron_learn(inputs, outputs_or, np.array([-0.2, 0.4]), 0.2, 0)

print("-----------------------------------------------")
print("AND gate")
perceptron_learn(inputs, outputs_and, np.array([0.2, 0.4]), 0.2, 0)

print("-----------------------------------------------")
print("NOT gate")
perceptron_learn(inputs_not, outputs_not, np.array([-0.3]), 0.2, 0)
