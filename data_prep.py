import pandas as pd
import argparse
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/raw/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/prepared/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()


def to_categorical(df: pd.DataFrame):
    df.MO = pd.Categorical(df.MO)
    df = df.assign(MO=df.MO.cat.codes)
    return df


def clean_data(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    df = df1.merge(df2, left_on='STA', right_on='WBAN')
    df.drop("STA", axis=1, inplace=True)
    df.drop("Date", axis=1, inplace=True)
    df.drop("Precip", axis=1, inplace=True)
    df.drop("WindGustSpd", axis=1, inplace=True)
    df.drop("MaxTemp", axis=1, inplace=True)
    df.drop("MinTemp", axis=1, inplace=True)
    df.drop("Snowfall", axis=1, inplace=True)
    df.drop("PoorWeather", axis=1, inplace=True)
    df.drop("YR", axis=1, inplace=True)
    df.drop("DA", axis=1, inplace=True)
    df.drop("PRCP", axis=1, inplace=True)
    df.drop("DR", axis=1, inplace=True)
    df.drop("SPD", axis=1, inplace=True)
    df.drop("MAX", axis=1, inplace=True)
    df.drop("MIN", axis=1, inplace=True)
    df.drop("MEA", axis=1, inplace=True)
    df.drop("SNF", axis=1, inplace=True)
    df.drop("SND", axis=1, inplace=True)
    df.drop("FT", axis=1, inplace=True)
    df.drop("FB", axis=1, inplace=True)
    df.drop("FTI", axis=1, inplace=True)
    df.drop("ITH", axis=1, inplace=True)
    df.drop("PGT", axis=1, inplace=True)
    df.drop("TSHDSBRSGF", axis=1, inplace=True)
    df.drop("SD3", axis=1, inplace=True)
    df.drop("RHX", axis=1, inplace=True)
    df.drop("RHN", axis=1, inplace=True)
    df.drop("RVG", axis=1, inplace=True)
    df.drop("WTE", axis=1, inplace=True)
    df.drop("WBAN", axis=1, inplace=True)
    df.drop("NAME", axis=1, inplace=True)
    df.drop("STATE/COUNTRY ID", axis=1, inplace=True)
    df.drop("LAT", axis=1, inplace=True)
    df.drop("LON", axis=1, inplace=True)
    df.drop("ELEV", axis=1, inplace=True)
    df.drop("Longitude", axis=1, inplace=True)
    df = to_categorical(df)
    return df


if __name__ == '__main__':
    args = parser_args_for_sac()
    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['data_preparation']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)

    df_wsum = pd.read_csv('..\\ML\\data\\raw\\Summary_of_Weather.csv', low_memory=False)
    df_sloc = pd.read_csv('..\\ML\\data\\raw\\Weather_Station_Locations.csv')

    #df_wsum = pd.read_csv('{input_dir}Summary_of_Weather.csv')
    #df_wsum = pd.read_csv(weather_summary)
    #df_sloc = pd.read_csv('{input_dir}Weather_Station_Locations.csv')
    #df_sloc = pd.read_csv(station_location)
    cleaned_data = clean_data(df1=df_wsum, df2=df_sloc)
    X, y = cleaned_data.drop("MeanTemp", axis=1), cleaned_data['MeanTemp']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=params.get('train_test_ratio'),
                                                        random_state=params.get('random_state'))
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                        train_size=params.get('train_val_ratio'),
                                                        random_state=params.get('random_state'))
    X_full_name = output_dir / 'X_full.csv'
    y_full_name = output_dir / 'y_full.csv'
    X_train_name = output_dir / 'X_train.csv'
    y_train_name = output_dir / 'y_train.csv'
    X_test_name = output_dir / 'X_test.csv'
    y_test_name = output_dir / 'y_test.csv'
    X_val_name = output_dir / 'X_val.csv'
    y_val_name = output_dir / 'y_val.csv'

    X.to_csv(X_full_name, index=False)
    y.to_csv(y_full_name, index=False)
    X_train.to_csv(X_train_name, index=False)
    y_train.to_csv(y_train_name, index=False)
    X_test.to_csv(X_test_name, index=False)
    y_test.to_csv(y_test_name, index=False)
    X_val.to_csv(X_val_name, index=False)
    y_val.to_csv(y_val_name, index=False)