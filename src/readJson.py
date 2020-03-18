#%%
def readJSONFile(path):
    import json
    with open(path) as f:
        data = json.load(f)
    return data

