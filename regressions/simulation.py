
def trading_outcome(real_change, predicted_change, thresholds=[0.001, 0.005, 0.01, 0.02, 0.03]):
    traders = [DummyTrader(t) for t in thresholds]
    rate = 1
    for real, prediction in zip(real_change, predicted_change):
        for trader in traders:
            trader.react(rate, prediction)
        rate *= 1 + real

    results = [(trader.portfolio_value(rate) - 1) * 100 for trader in traders]
    rate_diff = (rate - 1) * 100
    print(f'Raised funds by:', ', '.join(map(lambda perc: f'{perc:.4f}%', results)))
    print(f'Actually holding stocks all the time could result in {rate_diff:.4f}% profit.')
    return max(results)


class DummyTrader:
    def __init__(self, threshold):
        self.money = 1
        self.stocks = 0
        self.threshold = threshold

    def react(self, rate, prediction):
        if prediction >= self.threshold:
            self.buy(rate)
        elif prediction <= -self.threshold:
            self.sell(rate)

    def buy(self, rate):
        self.money, self.stocks = 0, self.stocks + self.money / rate

    def sell(self, rate):
        self.money, self.stocks = self.money + self.stocks * rate, 0

    def portfolio_value(self, current_rate):
        return self.money + self.stocks * current_rate
