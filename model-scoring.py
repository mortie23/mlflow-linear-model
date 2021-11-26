# %%
import os
import requests
import numpy as np
import pandas as pd

dataset = pd.DataFrame({"X": [0.5]})
dataset
# %%


def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}


data_json = dataset.to_dict(orient='split') if isinstance(
    dataset, pd.DataFrame) else create_tf_serving_json(dataset)

# %%
data_json
'''
{'index': [0], 'columns': ['X'], 'data': [[0.5]]}
'''

# %%
