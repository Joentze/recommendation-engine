import time
from preprocess import preprocess
from build_forest import preprocess_dataframe
from datasketch import MinHash
import numpy as np
from configs import PERMUTATIONS, FOREST_NAME, CSV_PATH
import pickle

def predict(text, database, perms, num_results, forest):
    
    start_time = time.time()
    tokens = preprocess(text)
    m = MinHash(num_perm=perms)

    for s in tokens:
        m.update(s.encode('utf8'))
        
    idx_array = np.array(forest.query(m, num_results))
    
    if len(idx_array) == 0:
        return None # if your query is empty, return none
    
    result = database.iloc[idx_array]['title']
    
    print('It took %s seconds to query forest.' %(time.time()-start_time))
    
    return result

def get_reco_by_ex_title(title, data, forest, no_of_recos):
  query_text = title + " " + data.loc[data["title"] == title, "description"]
  result = predict(query_text.values[0], data, PERMUTATIONS, no_of_recos, forest)
  return result

if __name__ == "__main__":
    with open(FOREST_NAME,"rb") as r_file:
        LSHForest = pickle.load(r_file)
    data = preprocess_dataframe(CSV_PATH)
    print(get_reco_by_ex_title("Rosewood Estate Winery Riesling AF 2017", data, LSHForest, 5))