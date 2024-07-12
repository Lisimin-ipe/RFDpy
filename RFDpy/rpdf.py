import numpy as np

class RPDF:
    def __init__(self, target_occurrences, target_probability, error_margin):
        self.target_occurrences = target_occurrences
        self.target_probability = target_probability
        self.error_margin = error_margin

    def update_training_data(self, X, y):
        occurrences = np.zeros(len(y), dtype=int)
        for unique_value in np.unique(y):
            indices = np.where(y == unique_value)[0]
            count = len(indices)
            if count < self.target_occurrences:
                additional_needed = self.target_occurrences - count
                occurrences[indices] += 1
                for _ in range(additional_needed):
                    X = np.vstack([X, X[indices]])
                    y = np.hstack([y, y[indices]])
                    occurrences = np.hstack([occurrences, [self.target_occurrences] * len(indices)])
        return X, y

    def evaluate_predictions(self, y_test, y_pred):
        errors = np.abs(y_test - y_pred)
        satisfactory_indices = np.where(errors <= self.error_margin)[0]
        probabilities = np.random.rand(len(satisfactory_indices))
        final_indices = satisfactory_indices[probabilities < self.target_probability]
        return final_indices
