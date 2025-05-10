import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump, load
import yaml
import argparse
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt

def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--baseline_model', '-bm', type=str,
                        default='data/models/LinearRegression.joblib',
                        required=False, help='path to linear regression prod version')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    parser.add_argument('--model_name', '-mn', type=str, default='XGBoost', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()

if __name__ == '__main__':
    args = parser_args_for_sac()
    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['xgboost']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    baseline_model_path = Path(args.baseline_model)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_joblib_path = output_dir / (args.model_name + '.joblib')

    X_train_name = input_dir / 'X_train.csv'
    y_train_name = input_dir / 'y_train.csv'
    X_test_name = input_dir / 'X_test.csv'
    y_test_name = input_dir / 'y_test.csv'

    X_train = pd.read_csv(X_train_name)
    y_train = pd.read_csv(y_train_name)
    X_test = pd.read_csv(X_test_name)
    y_test = pd.read_csv(y_test_name)

    xgb_model = GridSearchCV(XGBRegressor(), params['grid_search'])
    xgb_model = xgb_model.fit(X_train, y_train)

    baseline_model = load(baseline_model_path)
    y_pred_baseline = np.squeeze(baseline_model.predict(X_test))

    predicted_values = np.squeeze(xgb_model.predict(X_test))
    baseline_mae = mean_absolute_error(y_test, y_pred_baseline)
    model_mae = mean_absolute_error(y_test, predicted_values)

    print(xgb_model.score(X_test, y_test))
    print("Best params: ", xgb_model.best_params_)
    print("Baseline MAE: ", baseline_mae)
    print("Model MAE: ", model_mae)

    y_min = y_train.values.min()
    y_max = y_train.values.max()

    print("In range: [", y_min, ";", y_max, "]")
    print("MAE in percents: ", model_mae*100/(y_max-y_min), "%")

    # # Plot feature importance
    # fig, ax = plt.subplots(1,1,figsize=(10, 6))
    # plot_importance(xgb_model.best_estimator_, height=0.7, ax=ax)
    # plt.savefig('images/xgbFeatImp.png')
    # plt.show()

    dump(xgb_model, output_model_joblib_path)