import pandas as pd
def read_data(csvSource):
    return pandas.read_csv(csvSource)
def get_wpbc():
    data = pd.read_fwf("wpbc.data")
get_wpbc()