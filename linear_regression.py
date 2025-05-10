import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
import seaborn as sns

from joblib import dump

LINEAR_MODELS_MAPPER = {'Ridge': Ridge,
                        'LinearRegression': LinearRegression}


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    parser.add_argument('--model_name', '-mn', type=str, default='LinearRegression', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_sac()
    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['linear_regression']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    model_name = params.get('model_name')

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_path = output_dir / (model_name + '.csv')
    output_model_joblib_path = output_dir / (model_name + '.joblib')

    X_train_name = input_dir / 'X_train.csv'
    y_train_name = input_dir / 'y_train.csv'
    X_test_name = input_dir / 'X_test.csv'
    y_test_name = input_dir / 'y_test.csv'

    X_train = pd.read_csv(X_train_name)
    y_train = pd.read_csv(y_train_name)
    X_test = pd.read_csv(X_test_name)
    y_test = pd.read_csv(y_test_name)

    reg = LINEAR_MODELS_MAPPER.get(model_name)().fit(X_train, y_train)

    y_mean = y_train.mean()
    y_std = y_train.std()
    y_pred_baseline = np.random.normal(y_mean, y_std, len(y_train))

    predicted_values = np.squeeze(reg.predict(X_test))
    baseline_mae = mean_absolute_error(y_train, y_pred_baseline)
    model_mae = mean_absolute_error(y_test, predicted_values)

    print(reg.score(X_test, y_test))
    print("Baseline MAE: ", baseline_mae)
    print("Model MAE: ", model_mae)

    y_min = y_test.values.min()
    y_max = y_test.values.max()

    print("In range: [", y_min, ";", y_max, "]")
    print("MAE in percents: ", model_mae*100/(y_max-y_min), "%")

    intercept = reg.intercept_.astype(float)
    coefficients = reg.coef_.astype(float)
    intercept = pd.Series(intercept, name='intercept')
    coefficients = pd.Series(coefficients[0], name='coefficients')
    print("intercept:", intercept)
    print("list of coefficients:", coefficients)
    columns = [x for x in range(len(coefficients))]
    out_model = pd.DataFrame([coefficients, intercept])
    out_model.to_csv(output_model_path, index=False)

    # # Plot feature weights distributions
    # feature_weights = coefficients
    # feature_weights.add(intercept)
    # feature_names = X_train.columns
    # feature_names.append(pd.Index(["intercept"]))
    # plt.figure(figsize=(8, 4))
    # bar = sns.barplot(x=feature_weights, y=feature_names, width=0.6, color='lightskyblue')
    # plt.xlim(-0.30, 0.20)
    # plt.xlabel('Weights')
    # plt.ylabel('Features')
    # for i in bar.containers:
    #     bar.bar_label(i,)
    # plt.savefig('images/linRegrWeights.png')
    # plt.show()

    dump(reg, output_model_joblib_path)