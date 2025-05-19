# %% [markdown]
# Objective: 
# 1. Scrape data from the internet [At the current stage, we are using a structured dataset instead of scraping]
# 2. Unify the format and column naming of the data
# 3. Synthesize the data into a single dataset
# 4. Save the data as a json file
# - following the format: {"idiom1": {"CH_meaning": "explanation", "derivation": "derivation"}, "idiom2": {"CH_meaning": "explanation", "derivation": "derivation"}, ...}

# %% [markdown]
# notes. 
# - For convenience, at this stage, we are considering using organized datasets: [Github-Xinhua](https://github.com/pwxcoo/chinese-xinhua) and [CC-CEDICT](https://www.mdbg.net/chinese/dictionary?page=cedict)
# - CC-CEDICT is a large Chinese-English dictionary dataset, which includes some idioms, but also contains a large number of non-idiomatic vocabulary. Due to the lack of criteria to filter out true idioms, we currently decided not to use this dataset.
# - Other data sources requiring scraping are not considered for now

# %% packages
import json
import pandas as pd
import json
import os

# %% Load Github-Xinhua dataset
def load_dataset_GithubXinhua(dataset_path="../data/collection/GithubXinhua.json"):
    """
    Load and process idiom dataset from Xinhua dictionary JSON file; source: https://github.com/pwxcoo/chinese-xinhua
    
    Args:
        dataset_path (str): Path to the JSON file
        
    Returns:
        dict: Dictionary containing idioms with format:
            {"idiom1": {"CH_meaning": "explanation", "derivation": "derivation"}, ...}
    """
    # Load JSON data
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check missing ratio for each field
    total_records = len(data)
    missing_stats = {}

    for field in data[0].keys():
        missing_count = sum(1 for item in data if field not in item or not item[field])
        missing_ratio = missing_count / total_records
        missing_stats[field] = missing_ratio

    print("Missing ratio for each field:")
    for field, ratio in missing_stats.items():
        print(f"{field}: {ratio:.2%}")

    # Convert to desired dictionary format
    idioms_dict = {}
    for item in data:
        idiom = item['word']
        idioms_dict[idiom] = {
            'CH_meaning': item['explanation'],
            'derivation': item['derivation']
        }
    
    return idioms_dict

# Example usage
idioms_dict = load_dataset_GithubXinhua("../data/collection/GithubXinhua.json")
print(f"The Github-Xinhua dataset has {len(idioms_dict)} idioms")
print(list(idioms_dict.items())[:5])

# %% Save the dataset as json
with open("../data/collection/idiom.json", "w", encoding="utf-8") as f:
    json.dump(idioms_dict, f, ensure_ascii=False, indent=2)
print(f"The dataset has been saved to {os.path.abspath('../data/collection/idiom.json')}")
print(f"The dataset has {len(idioms_dict)} idioms")


