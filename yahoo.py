import sys
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError

def fetch_yahoo():
    my_share = share.Share('^HSI')
    symbol_data = None

    try:
        symbol_data = my_share.get_historical(share.PERIOD_TYPE_DAY,
                                              0,
                                              share.FREQUENCY_TYPE_DAY,
                                              5)
    except YahooFinanceError as e:
        print(e.message)
        sys.exit(1)

    new_x = {

        'Open': [],
        'High': [],
        'Low': []

    }
    new_x['Open'].append(symbol_data.get('open')[0])
    new_x['High'].append(symbol_data.get('high')[0])
    new_x['Low'].append(symbol_data.get('low')[0])
    return new_x

