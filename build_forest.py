import pickle
import pandas as pd

def preprocess_dataframe(filename:str):
    data = pd.read_csv(filename)
    data["text"] = data["title"] + " " +data["description"]
    data = data[data["text"].notna()]
    data = data.drop_duplicates()
    return data

def save_forest_obj_in_bin(pickle_filename:str, forest_obj):
    with open(pickle_filename, "wb") as w_file:
        pickle.dump(forest_obj, w_file)

# def build_forest_file(filename:str, pickle_filename: str):
#     data = preprocess_dataframe(filename)
#     forest = get_forest()

if __name__ == "__main__":
    pass