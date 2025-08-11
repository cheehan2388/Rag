import json

def calculate_accuracy(ground_truths, predictions):
    correct_count = 0
    total_count = len(ground_truths)

    # Create a dictionary to easily look up ground truth values by qid
    ground_truth_dict = {item['qid']: item['retrieve'] for item in ground_truths}

    # Iterate through predictions and compare with ground truth
    for prediction in predictions:
        qid = prediction['qid']
        predicted_value = prediction['retrieve']

        if qid in ground_truth_dict:
            if ground_truth_dict[qid] == predicted_value:
                correct_count += 1

    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0
    return accuracy

# Load JSON data from files
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        if isinstance(data, dict):
            return data['ground_truths'] if 'ground_truths' in data else data['answers']
        return data

# Example usage
ground_truths = load_json_file('ground_truths_example.json')
predictions = load_json_file('answers2.json')

accuracy = calculate_accuracy(ground_truths, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")