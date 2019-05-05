import json

from fbprophet import Prophet
import pandas as pd

from utils.config import Config

config = Config.open("../config.yml")


def predict_next(dataset: pd.DataFrame, begin: int, k: int = 3):
    m = Prophet()
    m.fit(dataset.iloc[:begin])
    future = m.make_future_dataframe(periods=k, include_history=False)
    return m.predict(future)


def _fmt_date(s):
    return s.replace('.', '-')


def run():
    full_table = pd.read_table(config("path.stocks"), ',')
    print(full_table.columns)
    table_fmt = pd.DataFrame(data={'ds': full_table['Date'].apply(_fmt_date), 'y': full_table.Close})

    with open(config("path.splits-tags"), "r") as splits_json:
        splits_tags = json.load(splits_json)
    to_predict = pd.Index(splits_tags['test']).intersection(full_table.Date)
    preds = {}
    for tag in to_predict[:5]:
        id = full_table[full_table.Date == tag].index[0]
        prediction_window = predict_next(table_fmt, id)
        preds[tag] = prediction_window.iloc[0]['yhat']

    pd.Series(preds).to_csv("tmp.csv")


if __name__ == '__main__':
    run()
