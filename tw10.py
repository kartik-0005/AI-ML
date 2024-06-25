import csv
import numpy as np

def load_data(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = np.array([[float(value) for value in row] for row in reader])
    return data[:, :-1], data[:, -1]

def calculate_priors(y):
    classes, counts = np.unique(y, return_counts=True)
    return dict(zip(classes, counts / len(y)))

def calculate_conditional_probabilities(X, y):
    conditional_probs = {}
    for cls in np.unique(y):
        X_cls = X[y == cls]
        feature_probs = [{val: count / len(X_cls) for val, count in zip(*np.unique(X_cls[:, col], return_counts=True))} for col in range(X.shape[1])]
        conditional_probs[cls] = feature_probs
    return conditional_probs

def classify(sample, priors, conditional_probs):
    probabilities = {}
    for cls, prior in priors.items():
        likelihood = np.prod([conditional_probs[cls][col].get(value, 1e-6) for col, value in enumerate(sample)])
        probabilities[cls] = prior * likelihood
    return max(probabilities, key=probabilities.get), probabilities

def evaluate(X, y, priors, conditional_probs):
    predictions = [classify(sample, priors, conditional_probs)[0] for sample in X]
    return np.mean(predictions == y)

def predict(input_features, priors, conditional_probs):
    prediction, probabilities = classify(input_features, priors, conditional_probs)
    for cls, prob in probabilities.items():
        print(f'P({cls}) = {prob:.6f}')
    return prediction

def main(file_path):
    X, y = load_data(file_path)
    X_discrete = np.floor(X)

    priors = calculate_priors(y)
    conditional_probs = calculate_conditional_probabilities(X_discrete, y)

    accuracy = evaluate(X_discrete, y, priors, conditional_probs)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    while True:
        input_str = input("\nEnter new sample (comma-separated values) or 'exit' to quit: ")
        if input_str.lower() == 'exit':
            break
        input_features = np.array([float(x) for x in input_str.split(',')])
        if len(input_features) != X.shape[1]:
            print(f"Please enter exactly {X.shape[1]} feature values.")
            continue
        input_features_discrete = np.floor(input_features)
        prediction = predict(input_features_discrete, priors, conditional_probs)
        print(f'Predicted class: {prediction}')

if __name__ == "__main__":
    main('Housing.csv')
