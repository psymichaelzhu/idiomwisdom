# %% [markdown]
# Objective: 
# > Analyze the final_idioms.json to answer the following questions:
# 1. What psychological categories or concepts are most frequently expressed through idioms? (category/concept frequency)
#       - answer: 
# 2. Which psychological categories or concepts are associated with greater internal diversity in idiomatic expression, potentially reflecting contextual dependence or conceptual ambiguity? (within category/concept: idiom meaning variance, probably through embedding)
#       - answer: 
# 3. Do certain types of psychological knowledge tend to be communicated through specific narrative forms, such as analogy, metaphor, or counterexample? (within category/concept: linguistic feature)
#       - answer: 
# 4. How have the usage patterns of idioms associated with different psychological categories or concepts shifted over history or across different contexts? (usage frequency/context)
#       - answer: 

# %% [markdown]
# notes. 
# - a

# %%packages
import json
import pandas as pd

# %% load data
# Load final_idioms.json
with open('../data/final_idioms.json', 'r', encoding='utf-8') as f:
    idioms_dict = json.load(f)

# Load embeddings.csv 
embeddings_df = pd.read_csv('../data/annotation/embeddings.csv')

# Print basic info
print(f"Number of idioms loaded: {len(idioms_dict)}")
print(f"Number of embeddings loaded: {len(embeddings_df)}")

# %%
# Check for duplicate names in embeddings
print("\nChecking for duplicates in embeddings:")
print(f"Total embeddings: {len(embeddings_df)}")
print(f"Unique embeddings: {len(embeddings_df['idiom'].unique())}")
duplicates = embeddings_df[embeddings_df['idiom'].duplicated()]
if len(duplicates) > 0:
    print(f"Found {len(duplicates)} duplicates:")
    print(duplicates['idiom'].tolist())
else:
    print("No duplicates found in embeddings")

# Check overlap between embeddings and idioms
embedding_names = set(embeddings_df['idiom'].unique())
idiom_names = set(idioms_dict.keys())

print("\nChecking overlap between embeddings and idioms:")
print(f"Names in embeddings: {len(embedding_names)}")
print(f"Names in idioms: {len(idiom_names)}")
print(f"Names in both: {len(embedding_names.intersection(idiom_names))}")
print(f"Names only in embeddings: {len(embedding_names - idiom_names)}")
print(f"Names only in idioms: {len(idiom_names - embedding_names)}")

# %%
# Remove duplicates from embeddings_df, keeping the first occurrence
print("\nRemoving duplicates from embeddings...")
original_length = len(embeddings_df)
embeddings_df = embeddings_df.drop_duplicates(subset=['idiom'], keep='first')
print(f"Removed {original_length - len(embeddings_df)} duplicate rows")

# Save the deduplicated DataFrame back to CSV
embeddings_df.to_csv('../data/annotation/embeddings.csv', index=False)
print("Saved deduplicated embeddings to embeddings.csv")

# %%
