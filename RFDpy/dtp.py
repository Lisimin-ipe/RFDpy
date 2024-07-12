import numpy as np
from .evaluation_func import Evaluation_func

class DTP:
    def __init__(self, model_factory, initial_train_data, initial_test_data, x_threshold, initial_lr, lr_decay, initial_regularization, regularization_decay):
        self.model_factory = model_factory
        self.initial_train_data = initial_train_data.copy()
        self.initial_test_data = initial_test_data.copy()
        self.performance_evaluator = Evaluation_func(x_threshold)
        self.initial_lr = initial_lr
        self.lr_decay = lr_decay
        self.current_lr = initial_lr
        self.initial_regularization = initial_regularization
        self.regularization_decay = regularization_decay
        self.current_regularization = initial_regularization
        self.cycle_count = 0  # Record training cycle count
        self.all_predictions = []  # Store predictions for each cycle
        self.all_actuals = []  # Store actual values for each cycle
        self.bad_train_data = []  # Store data that still performs poorly

    def train_model(self, train_data):
        self.model = self.model_factory(self.current_regularization)
        X_train, y_train = zip(*train_data)
        self.model.fit(np.array(X_train), np.array(y_train))

    def evaluate(self, data):
        X, y = zip(*data)
        predictions = self.model.predict(np.array(X))
        self.performance_evaluator.f_values = []
        prediction_actual_pairs = []
        for yp, yt in zip(predictions, y):
            self.performance_evaluator.calculate_f(yp, yt)
            prediction_actual_pairs.append((yp, yt))
        return prediction_actual_pairs, self.performance_evaluator.f_values

    def update_data(self, train_data, test_data):
        # Evaluate training data and retain poorly performing data
        _, train_f_values = self.evaluate(train_data)
        bad_train_data = [train_data[i] for i, f in enumerate(train_f_values) if f != 0]

        # Evaluate test data and move poorly performing data to training set
        _, test_f_values = self.evaluate(test_data)
        moved_to_train_data = [test_data[i] for i, f in enumerate(test_f_values) if f != 0]
        new_test_data = [test_data[i] for i, f in enumerate(test_f_values) if f == 0]

        # Add poorly performing training and test data to the training set
        new_train_data = train_data + bad_train_data + moved_to_train_data

        return new_train_data, new_test_data, bad_train_data

    def run_evaluation_cycles(self, M, th_save_path=None):
        current_train_data = self.initial_train_data.copy()
        current_test_data = self.initial_test_data.copy()
        results = []

        for i in range(M):
            self.train_model(current_train_data)
            train_predictions_actuals, _ = self.evaluate(current_train_data)
            y_test_predictions_actuals, _ = self.evaluate(current_test_data)
            
            self.train_predictions_actuals = train_predictions_actuals  # Save training set predictions and actual values
            self.all_predictions.append([p for p, _ in y_test_predictions_actuals])
            self.all_actuals.append([a for _, a in y_test_predictions_actuals])
            
            current_train_data, current_test_data, bad_train_data = self.update_data(current_train_data, current_test_data)
            results.append({
                'cycle': i + 1,
                'training_size': len(current_train_data),
                'test_size': len(current_test_data),
                'bad_train_data_size': len(bad_train_data)
            })
            self.cycle_count += 1  # Update training cycle count

            # Print the number of poorly performing data in the training set for each cycle
            print(f"Cycle {i + 1}: bad train data size = {len(bad_train_data)}")

            # Dynamically adjust learning rate and regularization parameters (only for the first 50% of cycles)
            if i < M // 2:
                self.current_lr *= self.lr_decay
                self.current_regularization *= self.regularization_decay

        self.bad_train_data = bad_train_data  # Save data that still performs poorly

        # Check if a save path is provided, if so, save the specified cycle data
        if th_save_path is not None:
            cycle_to_save = int(input("Enter the cycle number to save: ")) - 1
            if 0 <= cycle_to_save < M:
                self.export_cycle_data_to_txt(th_save_path, cycle_to_save)
        
        return results

    def export_results_to_txt(self, filename, results):
        with open(filename, 'w') as f_results:
            for result in results:
                f_results.write(str(result) + "\n")

    def export_test_data_to_txt(self, filename):
        with open(filename, 'w') as f_test:
            for cycle, (predictions, actuals) in enumerate(zip(self.all_predictions, self.all_actuals), start=1):
                f_test.write(f"Cycle {cycle}:\n")
                for yp, yt in zip(predictions, actuals):
                    f_test.write(f"Predicted: {yp}, Actual: {yt}\n")

    def export_bad_train_data_to_txt(self, filename):
        with open(filename, 'w') as f_bad_train:
            f_bad_train.write("Final bad train data:\n")
            for yp, yt in self.bad_train_data:
                f_bad_train.write(f"Predicted: {yp}, Actual: {yt}\n")

    def export_cycle_data_to_txt(self, filename, cycle):
        with open(filename, 'w') as f_cycle:
            f_cycle.write(f"Cycle {cycle + 1} data:\n")
            predictions = self.all_predictions[cycle]
            actuals = self.all_actuals[cycle]
            for yp, yt in zip(predictions, actuals):
                f_cycle.write(f"Predicted: {yp}, Actual: {yt}\n")
