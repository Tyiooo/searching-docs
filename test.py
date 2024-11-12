from model.search_engine import SearchEngine
import pandas as pd


def load_data():
    df = pd.read_csv('model/data.csv')
    return df


queries = ['global warming']

df = load_data()
model = SearchEngine(text_column='text',  id_column='labels')
model.fit(df, perform_stem=False)


for query in queries:
    model.get_results(query)