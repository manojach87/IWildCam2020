import json
#%%
def readJSONFile(path):
    with open(path) as f:
        data = json.load(f)
    return data

