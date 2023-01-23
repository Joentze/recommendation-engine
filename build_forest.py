import pickle
import pandas as pd
from datasketch import MinHash, MinHashLSHForest
import time
from preprocess import preprocess
from configs import PERMUTATIONS, CSV_PATH, FOREST_NAME

def build_forest(data, perms:int):    
    start_time = time.time()    
    minhash = []

    for text in data['text']:
        tokens = preprocess(text)
        m = MinHash(num_perm=perms)
        for s in tokens:
            m.update(s.encode('utf8'))
        minhash.append(m)
        
    forest = MinHashLSHForest(num_perm=perms)
    
    for i,m in enumerate(minhash):
        forest.add(i,m)
        
    forest.index()
    
    print('It took %s seconds to build forest.' %(time.time()-start_time))
    
    return forest

def preprocess_dataframe(filename:str):
    data = pd.read_csv(filename)
    data["text"] = data["title"] + " " +data["description"]
    data = data[data["text"].notna()]
    data = data.drop_duplicates()
    return data

def save_forest_obj_in_bin(pickle_filename:str, forest_obj):
    with open(pickle_filename, "wb") as w_file:
        pickle.dump(forest_obj, w_file)

def build_forest_file(filename:str, pickle_filename: str):
    data = preprocess_dataframe(filename)
    forest = build_forest(data, PERMUTATIONS)
    save_forest_obj_in_bin(pickle_filename, forest)


if __name__ == "__main__":
    build_forest_file(CSV_PATH, FOREST_NAME)