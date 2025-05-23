#%%
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt

#%%
# Step 1: Load the CSV (assumes first column is idiom, rest are embedding dims)
df = pd.read_csv('../data/annotation/embeddings.csv')
random_seed = 42
#df= df.sample(1000, random_state=random_seed)
df = df.drop('model', axis=1)

idioms = df.iloc[:, 0].values
embeddings = df.iloc[:, 1:].values.astype(np.float32)

# Step 2: Normalize the embeddings
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings_normalized = embeddings / norms

# Step 3: Compute cosine similarity (dense)
print("Computing pairwise cosine similarity...")
similarity_matrix = cosine_similarity(embeddings_normalized)

# Step 4: Flatten to long-form DataFrame (upper triangle only to save space)
rows = []
n = len(idioms)
for i in tqdm(range(n)):
    for j in range(i + 1, n):
        sim = similarity_matrix[i, j]
        rows.append((idioms[i], idioms[j], sim))

# Step 5: Save results
df_result = pd.DataFrame(rows, columns=["idiom1", "idiom2", "cosine_similarity"])
df_result.to_csv("../data/annotation/idiom_cosine_similarity.csv", index=False)
print("✅ Saved to idiom_cosine_similarity.csv")
# %%
# Read the similarity data
sim_df = pd.read_csv("../data/annotation/idiom_cosine_similarity.csv")

def find_most_similar_and_dissimilar(target_idiom, sim_df, top_k=5):
    """Find the most similar and dissimilar idioms for a given target idiom
    
    Args:
        target_idiom (str): The idiom to analyze
        sim_df (pd.DataFrame): DataFrame containing pairwise similarities
        top_k (int): Number of similar/dissimilar idioms to return
        
    Returns:
        most_similar (pd.DataFrame): Top k most similar idioms and their similarities
        most_dissimilar (pd.DataFrame): Top k most dissimilar idioms and their similarities
    """
    # Get all rows where target_idiom appears in either column
    mask1 = sim_df['idiom1'] == target_idiom
    mask2 = sim_df['idiom2'] == target_idiom
    
    # Combine and standardize the format
    similarities = []
    
    # When target is in idiom1 column
    df1 = sim_df[mask1][['idiom2', 'cosine_similarity']]
    df1.columns = ['idiom', 'similarity']
    similarities.append(df1)
    
    # When target is in idiom2 column
    df2 = sim_df[mask2][['idiom1', 'cosine_similarity']]
    df2.columns = ['idiom', 'similarity']
    similarities.append(df2)
    
    # Combine all similarities
    all_sims = pd.concat(similarities)
    
    # Sort and get top/bottom k
    most_similar = all_sims.nlargest(top_k, 'similarity')
    most_dissimilar = all_sims.nsmallest(top_k, 'similarity')
    
    return most_similar, most_dissimilar

# Example usage
target_idiom = "坚强不屈"  # Can be changed to any idiom
most_similar, most_dissimilar = find_most_similar_and_dissimilar(target_idiom, sim_df)

print(f"\nMost similar idioms to '{target_idiom}':")
print(most_similar.to_string(index=False))

print(f"\nMost dissimilar idioms to '{target_idiom}':")
print(most_dissimilar.to_string(index=False))

# %%
def get_similarity_distribution(idiom, sim_df, bin_width=0.1):
    """Calculate distribution of similarities for a given idiom across bins
    
    Args:
        idiom (str): Target idiom to analyze
        sim_df (pd.DataFrame): DataFrame containing pairwise similarities
        bin_width (float): Width of similarity bins
        
    Returns:
        dict: Dictionary mapping bin ranges to counts of idioms in that range
    """
    # Get all similarities for this idiom
    mask1 = sim_df['idiom1'] == idiom
    mask2 = sim_df['idiom2'] == idiom
    
    similarities = []
    similarities.extend(sim_df[mask1]['cosine_similarity'].tolist())
    similarities.extend(sim_df[mask2]['cosine_similarity'].tolist())
    
    # Create bins from 0 to 1 with specified width
    bins = np.arange(0, 1 + bin_width, bin_width)
    
    # Count items in each bin
    hist, _ = np.histogram(similarities, bins=bins)
    
    # Create dictionary mapping bin ranges to counts
    dist = {}
    for i in range(len(hist)):
        bin_range = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
        dist[bin_range] = hist[i]
        
    return dist


def plot_similarity_distributions(sim_df):
    """Plot similarity distributions for all idioms
    
    Args:
        sim_df (pd.DataFrame): DataFrame containing pairwise similarities
    """
    # Get unique idioms
    idioms = pd.unique(sim_df[['idiom1', 'idiom2']].values.ravel())
    
    # Calculate distributions for each idiom
    all_distributions = {}
    for idiom in tqdm(idioms):
        dist = get_similarity_distribution(idiom, sim_df)
        all_distributions[idiom] = dist
        
    # Convert to DataFrame for easier plotting
    df_dist = pd.DataFrame(all_distributions).T
    
    # Calculate mean distribution
    mean_dist = df_dist.mean()
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Plot individual distributions with low alpha
    for idx in df_dist.index:
        plt.plot(range(len(df_dist.columns)), df_dist.loc[idx], 
                color='gray', alpha=0.1, linewidth=1)
    
    # Plot mean distribution
    plt.plot(range(len(mean_dist)), mean_dist, 
            color='blue', linewidth=3, label='Mean Distribution')
    
    # Customize plot
    plt.title('Distribution of Similarity Scores Across Idioms', fontsize=22)
    plt.xlabel('Similarity Score Bins', fontsize=18)
    plt.ylabel('Count of Idioms', fontsize=18)
    plt.xticks(range(len(df_dist.columns)), 
               df_dist.columns, 
               rotation=45, 
               fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    
    plt.tight_layout()
    plt.show()

# Example usage
plot_similarity_distributions(sim_df)

# %%
import faiss
import numpy as np

X = np.load('idiom_embeddings.npy').astype('float32')
faiss.normalize_L2(X)  # inplace normalization

index = faiss.IndexFlatIP(X.shape[1])
index.add(X)

D, I = index.search(X, k=10)  # D: similarity, I: index of top-k


#%%
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

df = pd.read_csv('../data/annotation/embeddings.csv')
df = df.drop('model', axis=1)
idioms = df.iloc[:, 0].values
X = df.iloc[:, 1:].values.astype(np.float32)

# Normalize
norms = np.linalg.norm(X, axis=1, keepdims=True)
X = X / norms

batch_size = 1000
n = X.shape[0]

with open("../data/annotation/idiom_cosine_similarity.csv", "w") as f:
    f.write("idiom1,idiom2,cosine_similarity\n")

    for i in tqdm(range(0, n, batch_size)):
        end_i = min(i + batch_size, n)
        X_block = X[i:end_i]
        sim_block = np.dot(X_block, X.T)

        for ii in range(X_block.shape[0]):
            for j in range(i + ii + 1, n):  # 保证是上三角
                idiom1 = idioms[i + ii]
                idiom2 = idioms[j]
                sim = sim_block[ii, j]
                f.write(f"{idiom1},{idiom2},{sim:.6f}\n")
# %%
import pandas as pd
import numpy as np
import dask.array as da
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import os

# Step 1: Load data
df = pd.read_csv('../data/annotation/embeddings.csv')
df = df.drop('model', axis=1)

idioms = df.iloc[:, 0].values
embeddings_np = df.iloc[:, 1:].values.astype(np.float32)

# Step 2: Convert to Dask array and normalize
embeddings = da.from_array(embeddings_np, chunks=(1000, -1))
norms = da.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings_normalized = embeddings / norms

# Step 3: Compute cosine similarity matrix
similarity_matrix = embeddings_normalized @ embeddings_normalized.T

print(similarity_matrix)
# %%
# Step 4: Only take upper triangle (excluding diagonal)
n = embeddings.shape[0]
i_idx, j_idx = da.triu_indices(n, k=1)

# Step 5: Map indices to idioms and extract similarities
idiom1_array = da.from_array(idioms, chunks=(1000,))[i_idx]
idiom2_array = da.from_array(idioms, chunks=(1000,))[j_idx]
similarity_array = similarity_matrix[i_idx, j_idx]

# Step 6: Combine into Dask DataFrame
ddf = dd.from_dask_array(
    da.stack([idiom1_array, idiom2_array, similarity_array], axis=1),
    columns=["idiom1", "idiom2", "cosine_similarity"]
)

# Step 7: Save to Parquet
output_path = "../data/annotation/idiom_similarity_parquet"
os.makedirs(output_path, exist_ok=True)

with ProgressBar():
    ddf.to_parquet(output_path, engine='pyarrow', write_index=False)

print(f"✅ Saved similarity matrix to: {output_path}")
# %% compute similarity | CORE
import pandas as pd
import numpy as np
import dask.array as da
import time

chunk_size = 2500 # Design 1: meidium size; too small x too large x

# Record total start time
total_start = time.time()

# Step 1: Load data
t1 = time.time()
df = pd.read_parquet('../data/annotation/embeddings.parquet') # Design 2: parquet instead of csv

# preprocess
df = df.drop('model', axis=1)
idioms = df.iloc[:, 0].values
embeddings_np = df.iloc[:, 1:].values.astype(np.float32)
t2 = time.time()

# Step 2: Convert to Dask array and normalize
t3 = time.time()
embeddings = da.from_array(embeddings_np, chunks=(chunk_size, -1))
norms = da.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings_normalized = (embeddings / norms).persist() # Design 3: persist() to avoid recomputation
t4 = time.time()

# Step 3: Compute cosine similarity matrix
t5 = time.time()
similarity_matrix = da.dot(embeddings_normalized, embeddings_normalized.T)
similarity_matrix_result = similarity_matrix.compute()
t6 = time.time()

# Record total end time
total_end = time.time()

# Report timing
print(f"⏱️ Timing Results | Chunk Size: {chunk_size}")
print(f"Step 1 - Load Data + Preprocess: {t2 - t1:.3f} sec")
print(f"Step 2 - Normalize with Dask:    {t4 - t3:.3f} sec")
print(f"Step 3 - Compute Similarity:     {t6 - t5:.3f} sec")
print(f"✅ Total Runtime:                {total_end - total_start:.3f} sec")


# %% Save cosine similarity matrix | CORE
import pandas as pd
import numpy as np
import os
from dask import delayed, compute
from tqdm import tqdm
import time

similarity_output_path = "../data/embedding/similarity"
if not os.path.exists(similarity_output_path):
    os.makedirs(similarity_output_path)
n_chunks = 20 # no more than 100

# Record total start time
save_start = time.time()


# Step 1: Generate index pairs
t1 = time.time()
n = len(idioms)
i_idx, j_idx = np.triu_indices(n, k=1)  # use upper triangle to avoid duplicates
t2 = time.time()

# Step 2: Construct delayed save tasks by chunk
chunk_size = int(np.ceil(len(i_idx) / n_chunks))
tasks = []

# construct delayed save tasks
@delayed
def build_and_save_chunk(i_range, j_range, chunk_id, output_path=similarity_output_path):
    # extract idioms and scores
    idiom1 = idioms[i_range]
    idiom2 = idioms[j_range]
    scores = similarity_matrix_result[i_range, j_range]
    
    # build dataframe
    df = pd.DataFrame({'idiom1': idiom1, 'idiom2': idiom2, 'cosine': scores})
    
    # save to parquet    
    path = f"{output_path}/chunk_{chunk_id:03d}.parquet"  # Use sequential chunk IDs with zero padding
    df.to_parquet(path, index=False, compression='snappy')
    return path

for chunk_id, start in enumerate(range(0, len(i_idx), chunk_size)):
    i_chunk = i_idx[start:start + chunk_size]
    j_chunk = j_idx[start:start + chunk_size]
    tasks.append(build_and_save_chunk(i_chunk, j_chunk, chunk_id))
t3 = time.time()

# Step 3: Combine and execute tasks
@delayed
def finalize(paths):
    print(f"All {len(paths)} chunks written")

final_task = finalize(tasks)
final_task.visualize(filename=f"{similarity_output_path}/full_dag", format="png")
compute(final_task)
t4 = time.time()

# Final timing report
save_end = time.time()
print("⏱️ Timing Results for Saving Cosine Similarity")
print(f"Step 1 - Generate index pairs:     {t2 - t1:.3f} sec")
print(f"Step 2 - Build delayed tasks:      {t3 - t2:.3f} sec")
print(f"Step 3 - Execute save tasks:       {t4 - t3:.3f} sec")
print(f"✅ Total Save Time:                {save_end - save_start:.3f} sec")

#%%
import dask.array as da
import dask.dataframe as dd

# similarity_matrix_result: Dask Array (normalized cosine similarity)
n = len(idioms)
pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
idiom1 = [idioms[i] for (i, j) in pairs]
idiom2 = [idioms[j] for (i, j) in pairs]
scores = [similarity_matrix_result[i, j] for (i, j) in pairs]  # lazy list

df = dd.from_pandas(pd.DataFrame({'idiom1': idiom1, 'idiom2': idiom2, 'cosine': scores}), npartitions=50)
df.to_parquet("../data/annotation/idiom_sim.parquet", compression='snappy')


#%% save to parquet ｜CORE
from dask import delayed, compute
import dask.dataframe as dd
from tqdm import tqdm

# step 1: construct delayed tasks
@delayed
def build_and_save_chunk(i_range, j_range, output_path="../data/embedding/similarity"):
    # extract idioms and scores
    idiom1 = idioms[i_range]
    idiom2 = idioms[j_range]
    scores = similarity_matrix_result[i_range, j_range]
    # build dataframe
    df = pd.DataFrame({'idiom1': idiom1, 'idiom2': idiom2, 'cosine': scores})
    # save to parquet
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    path = f"{output_path}/chunk_{i_range[0]}.parquet"
    df.to_parquet(path, index=False, compression='snappy')
    return path

tasks = []
chunk_size = 1000000

# step 2: compute indices
n = len(idioms)
i_idx, j_idx = np.triu_indices(n, k=1) # use upper triangle to avoid duplicate

# step 3: employ delayed tasks by chunk
for start in range(0, len(i_idx), chunk_size):
    i_chunk = i_idx[start:start+chunk_size]
    j_chunk = j_idx[start:start+chunk_size]
    tasks.append(build_and_save_chunk(i_chunk, j_chunk))
# step 4: compute
compute(*tasks)



#%%
import dask.array as da
import dask.dataframe as dd
from tqdm import tqdm

# 假设 similarity_matrix 是 shape=(n, n) 的 NumPy 数组
n = similarity_matrix.shape[0]
i_idx, j_idx = np.triu_indices(n, k=1)
chunk_size = 5000
for start in tqdm(range(0, len(i_idx), chunk_size)):
    end = start + chunk_size
    chunk = pd.DataFrame({
        "idiom1": [idioms[i] for i in i_idx[start:end]],
        "idiom2": [idioms[j] for j in j_idx[start:end]],
        "cosine": similarity_matrix[i_idx[start:end], j_idx[start:end]]
    })
    chunk.to_parquet(f"../data/annotation/idiom_sim_{start//chunk_size}.parquet", compression='snappy')

#%%
import pandas as pd
from tqdm import tqdm

rows = []
n = len(idioms)

for i in tqdm(range(n)):
    for j in range(i + 1, n):
        sim = similarity_matrix_result[i, j]
        rows.append((idioms[i], idioms[j], float(sim)))

df_sim = pd.DataFrame(rows, columns=["idiom1", "idiom2", "cosine_similarity"])
df_sim.to_parquet("../data/annotation/idiom_similarities.parquet")

#%%
# Test different chunk sizes and plot timing results
import matplotlib.pyplot as plt

def compute_similarity_timing(chunk_size, embeddings_np):
    """
    Compute similarity matrix timing for a given chunk size
    
    Input:
        chunk_size(int): size of chunks for dask array
        embeddings_np(np.array): numpy array of embeddings
        
    Output:
        timing(dict): dictionary containing timing results for each step
    """
    # Convert to Dask array and normalize
    t1 = time.time()
    embeddings = da.from_array(embeddings_np, chunks=(chunk_size, -1))
    norms = da.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / norms
    t2 = time.time()
    normalize_time = t2 - t1
    
    # Compute similarity matrix
    t3 = time.time()
    similarity_matrix = da.dot(embeddings_normalized, embeddings_normalized.T)
    similarity_matrix_result = similarity_matrix.compute()
    t4 = time.time()
    compute_time = t4 - t3
    
    return {'normalize': normalize_time, 'compute': compute_time, 'total': normalize_time + compute_time}

# Test different chunk sizes
chunk_sizes = [5000, 10000, 15000, 20000, 25000]
timings = []

for cs in chunk_sizes:
    print(f"Testing chunk size: {cs}")
    timing = compute_similarity_timing(cs, embeddings_np)
    timings.append(timing)

# Plot results
plt.figure(figsize=(10, 6))

normalize_times = [t['normalize'] for t in timings]
compute_times = [t['compute'] for t in timings]
total_times = [t['total'] for t in timings]

plt.plot(chunk_sizes, normalize_times, 'o-', label='Normalize Time', color='#1b9e77')
plt.plot(chunk_sizes, compute_times, 'o-', label='Compute Time', color='#d95f02')
plt.plot(chunk_sizes, total_times, 'o-', label='Total Time', color='#7570b3')

plt.xlabel('Chunk Size', fontsize=18)
plt.ylabel('Time (seconds)', fontsize=18)
plt.title('Similarity Matrix Computation Time\nby Chunk Size', fontsize=22)
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.tight_layout()
plt.show()














# %%
# Load embeddings CSV and save as parquet
import pandas as pd
import os

# Load embeddings CSV
df = pd.read_csv('../data/annotation/embeddings.csv')

# Create output directory if it doesn't exist
output_path = '../data/annotation'
os.makedirs(output_path, exist_ok=True)

# Save as parquet
df.to_parquet(f'{output_path}/embeddings.parquet', engine='pyarrow', index=False)

print(f"✅ Saved embeddings to: {output_path}")
# %%
# 假设你已定义：
# similarity_matrix = embeddings_normalized @ embeddings_normalized.T

similarity_matrix.visualize(rankdir="BT")  # BT = Bottom-Top
# %%
A = da.from_array(embeddings_np[:3000], chunks=(1000, -1))
A.visualize(rankdir="BT")
# %%
