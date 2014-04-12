

import json

filename = './data/data.json'

with open(filename, 'r') as f:
    tweets = json.load(f)