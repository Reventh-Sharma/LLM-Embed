from sklearn.metrics import precision_score, recall_score, f1_score


class EmbeddingModelMetrics:
    def __init__(self, true_labels, predicted_labels):
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels

    def calculate_precision(self):
        return precision_score(self.true_labels, self.predicted_labels, average='weighted')

    def calculate_recall(self):
        return recall_score(self.true_labels, self.predicted_labels, average='weighted')

    def calculate_f1_score(self):
        return f1_score(self.true_labels, self.predicted_labels, average='weighted')

# Toy data for testing
true_labels = [1, 0, 1, 1, 0, 1, 0, 1, 0, 0]
predicted_labels = [1, 1, 0, 1, 0, 0, 1, 1, 0, 0]

# Create an instance of the EmbeddingModelMetrics class
metrics_calculator = EmbeddingModelMetrics(true_labels, predicted_labels)

# Calculate and print precision, recall, and F1-score
precision = metrics_calculator.calculate_precision()
recall = metrics_calculator.calculate_recall()
f1_score = metrics_calculator.calculate_f1_score()

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
