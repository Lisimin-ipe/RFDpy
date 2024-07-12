# RFDpy

RFDpy is a Python package for machine learning model evaluation and data processing. It includes tools for dynamic training processing (DTP) and robust prediction data filtering (RPDF).

## Installation

You can install the package using pip:

```sh
pip install RFDpy


Usage

from RFDpy import DTP, RPDF
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52)
train_data = list(zip(X_train, y_train))
test_data = list(zip(X_test, y_test))

# Example usage of DTP
def model_factory(regularization):
    return XGBRegressor(n_estimators=100, random_state=52,alpha=regularization)

evaluator = DTP(
    model_factory=model_factory,
    initial_train_data=train_data,
    initial_test_data=test_data,
    x_threshold=0.05,
    initial_lr=0.01,
    lr_decay=0.9,
    initial_regularization=0.001,
    regularization_decay=0.9
)
results = evaluator.run_evaluation_cycles(M=10, th_save_path='path\\to\\save\\cycle_data.txt')


for result in results:
    print(result)

evaluator.export_results_to_txt('path\\to\\save\\evaluation_results.txt', results)
evaluator.export_data_to_txt('path\\to\\save\\detailed_data.txt')

# Example usage of RPDF
rpdf = RPDF(target_occurrences=100, target_probability=0.95, error_margin=0.05)
X_train_updated, y_train_updated = rpdf.update_training_data(X_train, y_train)

xgb_regr = XGBRegressor(n_estimators=100, random_state=52)
xgb_regr.fit(X_train_updated, y_train_updated)
y_pred = xgb_regr.predict(X_test)

satisfactory_indices = rpdf.evaluate_predictions(y_test, y_pred)
satisfactory_values = y_test[satisfactory_indices]

output_file_path = 'path\\to\\save\\output_file.txt'
with open(output_file_path, 'w') as file:
    for value in satisfactory_values:
        file.write(f"{value}\n")

print(f"Filtered true values saved to: {output_file_path}")