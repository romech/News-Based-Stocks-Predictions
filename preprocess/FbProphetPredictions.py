from functools import lru_cache as cached
from multiprocessing import Pool

import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from fbprophet import Prophet
from utils.config import Config

config = Config.open("../config.yml")


def run():
    df = pd.read_csv(config("path.stocks"))
    prep = df[['Date', 'Close']]. \
        rename(columns={'Date': 'ds', 'Close': 'y'}). \
        sort_values(by='ds'). \
        reset_index(drop=True)

    @cached()
    def predict_date(target):
        date_index, date = target
        train_data = prep.iloc[:date_index]
        m = Prophet(daily_seasonality=False)
        m.fit(train_data)
        out = m.make_future_dataframe(periods=10, include_history=False)
        prediction = m.predict(out)
        return prediction[prediction.ds == date][['yhat', 'yhat_lower', 'yhat_upper']]

    with Pool(processes=8) as pool:
        prediction_rows = pool.map(predict_date, [(i, row.ds) for i, row in prep.iloc[252:].iterrows()])

    prophecies = pd.concat(prediction_rows)
    df_union = pd.concat([prep.iloc[252:].reset_index(drop=True), prophecies], sort=False, axis=1)
    df_union.rename({'ds': 'Date',
                     'y': 'Actual',
                     'yhat': 'Predicted',
                     'yhat_lower': 'Lower',
                     'yhat_upper': 'Upper'},
                    axis='columns'). \
        to_csv(config("path.fbprophet"), index=False)


if __name__ == '__main__':
    run()
