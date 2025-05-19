# %% [markdown]
# Objective:
# 1. Compare the quality of each model embedding through a benchmark task
# 2. Validate the idiom wholistic embedding
# 3. Derive embeddings for each idiom

#%% [markdown]
# Models:
#   - Whole-word-matching Chinese  BERT: https://github.com/ymcui/Chinese-BERT-wwm
#       - extended version: BERT-wwm-ext
#       - large RoBERTa version: RoBERTa-wwm-ext-large
#   - Light Chinese BERT (mengzi): https://github.com/Langboat/Mengzi/blob/main/README_en.md
#   - Chinese BERT (google): https://huggingface.co/google-bert/bert-base-chinese

#%% packages
from scipy.spatial.distance import cosine
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import zhplot
import json
import os
from tqdm import tqdm
import pandas as pd

# %% helper functions
def calculate_similarities(target_emb, word_list, model_instance):
        """Calculate cosine similarities between target embedding and a list of words
        
        Args:
            target_emb (np.array): Target word embedding
            word_list (List[str]): List of words to compare against
            model_instance (object): Model instance to get embeddings
            
        Returns:
            similarities (List[float]): List of cosine similarities rounded to 3 decimal places
        """
        similarities = []
        for word in word_list:
            word_emb = model_instance.get_embedding(word).detach().numpy().squeeze()
            sim = round(1 - cosine(target_emb, word_emb), 3)
            similarities.append(sim)
        return similarities

def embedding_benchmark(model_instance, word_dict):
    """Test semantic relationships between words using embeddings and cosine similarity
    
    Args:
        model_instance (object): Instance of embedding model class with get_embedding method
        word_dict (dict): Dictionary containing target word and related words
                         Format: {'idiom': str, 'synonyms': List[str], 'antonyms': List[str], 'baseline': List[str]}
    
    Returns:
        sims (dict): Dictionary containing similarity scores
                      Format: {'synonyms': List[float], 'antonyms': List[float], 'baseline': List[float]}
    """
    # Get target word embedding
    target_emb = model_instance.get_embedding(word_dict['idiom']).detach().numpy().squeeze()
    
    # Create a copy of word_dict without modifying the original
    word_lists = {k: v for k, v in word_dict.items() if k != 'idiom'}
    
    # Calculate similarities for all word lists
    sims = {}
    for key, word_list in word_lists.items():
        sims[key] = calculate_similarities(target_emb, word_list, model_instance)
    return sims


def compare_benchmark(instance1, instance2, benchmark_list):
    """Compare embeddings from two BERT models on a benchmark dataset
    
    Args:
        instance1 (BertEmbedding): First BERT model instance
        instance2 (BertEmbedding): Second BERT model instance 
        benchmark_list (list): List of dictionaries containing benchmark data with idioms and related phrases
        
    Returns:
        None: Prints comparison results for each benchmark item
    """
    import matplotlib.pyplot as plt
    import numpy as np
    # Store all results for final average
    all_results1 = []
    all_results2 = []
    
    for data in benchmark_list:
        # Create a deep copy of data for each model to avoid modifying original
        data1 = {k: v for k, v in data.items()}
        data2 = {k: v for k, v in data.items()}
        
        # Get results
        result1 = embedding_benchmark(instance1, data1)
        result2 = embedding_benchmark(instance2, data2)
        
        # Store results
        all_results1.append(result1)
        all_results2.append(result2)
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(f"Idiom: {data['idiom']}", fontsize=22)
        
        # Prepare data for both models - reorder categories
        categories = ['Synonyms', 'Baseline', 'Antonyms']
        cat_keys = ['synonyms', 'baseline', 'antonyms']
        
        # Calculate value ranges for consistent colorbar
        all_values = []
        for result in [result1, result2]:
            values = []
            for cat in cat_keys:
                values.extend(result[cat])
            all_values.extend(values)
        vmin, vmax = min(all_values), max(all_values)
        
        for ax, result, model in [(ax1, result1, instance1), (ax2, result2, instance2)]:
            # Get values and calculate means
            values = []
            for cat in cat_keys:
                values.append(result[cat])
            values = np.array(values)
            means = np.mean(values, axis=1)
            
            # Add mean column to values
            values_with_mean = np.column_stack((values, means))
            
            # Create heatmap with red-white-blue colormap and consistent scale
            im = ax.imshow(values_with_mean, cmap='RdBu', vmin=vmin, vmax=vmax)
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
            
            # Add labels
            ax.set_title(f"Model: {model.model.config._name_or_path}", fontsize=18)
            ax.set_yticks(range(len(categories)))
            ax.set_yticklabels(categories, fontsize=18)
            
            # X-axis labels (indices + "Mean")
            x_labels = [str(i+1) for i in range(values.shape[1])] + ['Mean']
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, fontsize=18)
            
            # Add text annotations
            for i in range(len(categories)):
                for j in range(values_with_mean.shape[1]):
                    ax.text(j, i, f'{values_with_mean[i, j]:.3f}',
                           ha='center', va='center', fontsize=18)
        
        plt.tight_layout()
        plt.show()
    
    # Create average plot across all idioms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Average Across All Idioms", fontsize=22)
    
    # Calculate overall value ranges for consistent colorbar
    all_avg_values = []
    for all_results in [all_results1, all_results2]:
        for cat in cat_keys:
            cat_values = [result[cat] for result in all_results]
            all_avg_values.extend(np.mean(cat_values, axis=0))
    vmin, vmax = min(all_avg_values), max(all_avg_values)
    
    for ax, all_results, model in [(ax1, all_results1, instance1), (ax2, all_results2, instance2)]:
        # Calculate average values
        avg_values = []
        for cat in cat_keys:
            cat_values = [result[cat] for result in all_results]
            avg_values.append(np.mean(cat_values, axis=0))
        avg_values = np.array(avg_values)
        means = np.mean(avg_values, axis=1)
        
        # Add mean column
        values_with_mean = np.column_stack((avg_values, means))
        
        # Create heatmap with red-white-blue colormap and consistent scale
        im = ax.imshow(values_with_mean, cmap='RdBu', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax)
        
        # Add labels
        ax.set_title(f"Model: {model.model.config._name_or_path}", fontsize=18)
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories, fontsize=18)
        
        x_labels = [str(i+1) for i in range(avg_values.shape[1])] + ['Mean']
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=18)
        
        # Add text annotations
        for i in range(len(categories)):
            for j in range(values_with_mean.shape[1]):
                ax.text(j, i, f'{values_with_mean[i, j]:.3f}',
                       ha='center', va='center', fontsize=18)
    
    plt.tight_layout()
    plt.show()

# %% BERT class
class BertEmbedding:
    """A parent class to generate embeddings using BERT models
    
    This class initializes a BERT model and tokenizer, and provides methods
    to generate embeddings. Child classes should specify the model name.
    """
    
    def __init__(self, model_name):
        """Initialize the BERT model and tokenizer
        
        Args:
            model_name (str): Name of the pretrained BERT model
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        
    def get_embedding(self, text):
        """Get embeddings for input text and return as a dataframe
        
        Args:
            text (str): Input text to get embedding for
            
        Returns:
            df (pd.DataFrame): DataFrame containing idiom, tokens and embedding dimensions
        """
        import pandas as pd
        
        # Get embeddings
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.squeeze().detach().numpy()
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Create dataframe
        data = []
        for token, emb in zip(tokens, embedding):
            row = [text, token] + list(emb)
            data.append(row)
            
        # Column names
        cols = ['idiom', 'token'] + [f'ebd_{i}' for i in range(embedding.shape[1])]
        
        # Create dataframe
        df = pd.DataFrame(data, columns=cols)
        
        return df
    
    def get_summarized_embedding(self, text):
        """Get summarized embedding for input text by returning only the CLS token embedding
        
        Args:
            text (str): Input text to get summarized embedding for
            
        Returns:
            embedding (torch.Tensor): CLS token embedding tensor
        """
        # Get embeddings
        inputs = self.tokenizer(text, return_tensors="pt")
        # Obtain the embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get only the CLS token embedding (first token)
        embedding = outputs.last_hidden_state[:,0,:].squeeze()
        return embedding

# mengzi
mengzi = BertEmbedding("Langboat/mengzi-bert-base")
a=mengzi.get_embedding("刻舟求剑")
print(a)
print(a.shape)

# wwm-ext
wwm = BertEmbedding("hfl/chinese-bert-wwm-ext")
a=wwm.get_embedding("刻舟求剑")
print(a.shape)

# wwm-large
wwm_large = BertEmbedding("hfl/chinese-roberta-wwm-ext-large")
a=wwm_large.get_embedding("刻舟求剑")
print(a.shape)

# basic
basic = BertEmbedding("bert-base-chinese")
a=basic.get_embedding("刻舟求剑")
print(a.shape)


#%% all embeddings
def generate_all_embeddings(model_instance, idiom_dict, save_dir='../data/embedding'):
    """Generate embeddings for all idioms in the dictionary and save to CSV
    
    Args:
        model_instance (BertEmbedding): BERT model instance to generate embeddings
        idiom_dict (dict): Dictionary containing idioms
        save_dir (str): Directory to save the output CSV file
    
    Output:
        CSV file saved to data/embedding/[model_name].csv containing embeddings for all idioms
    """
    import pandas as pd
    from tqdm import tqdm
    import os
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get model name from instance
    model_name = model_instance.model.config._name_or_path.replace('/', '_')
    save_path = os.path.join(save_dir, f'{model_name}.csv')
    
    # Check if file exists and load existing embeddings
    if os.path.exists(save_path):
        all_embeddings = pd.read_csv(save_path)
        # Get list of idioms already processed
        processed_idioms = all_embeddings['idiom'].unique()
    else:
        all_embeddings = pd.DataFrame()
        processed_idioms = []
    
    # Process each idiom with progress bar, skipping already processed ones
    for idiom in tqdm(idiom_dict.keys(), desc=f'Generating embeddings for {model_name}'):
        if idiom in processed_idioms:
            continue
            
        # Get embedding dataframe for current idiom
        idiom_df = model_instance.get_embedding(idiom)
        
        # Append to main dataframe
        all_embeddings = pd.concat([all_embeddings, idiom_df], ignore_index=True)
        
        # Save after each idiom in case of interruption
        all_embeddings.to_csv(save_path, index=False)

# Load idiom dictionary
with open('../data/collection/idiom.json', 'r', encoding='utf-8') as f:
    idiom_dict = json.load(f)

# Generate embeddings for each model
generate_all_embeddings(wwm, idiom_dict)
generate_all_embeddings(basic, idiom_dict)
generate_all_embeddings(wwm_large, idiom_dict)
generate_all_embeddings(mengzi, idiom_dict)





# %% benchmark comparison
benchmark_data=[
{
  "idiom": "鼠目寸光",
  "synonyms": ["鼠目寸光", "目光短浅", "急功近利"],
  "antonyms": ["高瞻远瞩", "深谋远虑", "未雨绸缪"],
  "baseline": ["花红柳绿", "杯水车薪", "如释重负"]
},
{
  "idiom": "杀鸡儆猴",
  "synonyms": ["杀鸡儆猴", "以儆效尤", "杀一儆百"],
  "antonyms": ["网开一面", "宽大为怀", "仁至义尽"],
  "baseline": ["牛头马面", "画蛇添足", "八面玲珑"]
},
{
  "idiom": "独善其身",
  "synonyms": ["独善其身", "洁身自好", "避世隐居"],
  "antonyms": ["兼济天下", "挺身而出", "赴汤蹈火"],
  "baseline": ["风和日丽", "指鹿为马", "骑虎难下"]
},
{
  "idiom": "纸上谈兵",
  "synonyms": ["纸上谈兵", "闭门造车", "空谈误国"],
  "antonyms": ["身临其境", "知行合一", "脚踏实地"],
  "baseline": ["对牛弹琴", "半信半疑", "火烧眉毛"]
},
{
  "idiom": "亡羊补牢",
  "synonyms": ["亡羊补牢", "痛定思痛", "吃一堑长一智"],
  "antonyms": ["重蹈覆辙", "执迷不悟", "一错再错"],
  "baseline": ["落英缤纷", "狼狈为奸", "鸡犬不宁"]
}
]

# Compare mengzi and wwm-ext models
#compare_benchmark(mengzi, wwm, benchmark_data)

# Compare basic and wwm-ext models
#compare_benchmark(basic, wwm, benchmark_data)

# Compare wwm and wwm-large models
#compare_benchmark(wwm, wwm_large, benchmark_data)


#%%
#%% all embeddings
def generate_all_embeddings(model_instance, idiom_dict, save_dir='../data/embedding/CLS'):
    """Generate embeddings for all idioms in the dictionary and save to CSV
    
    Args:
        model_instance (BertEmbedding): BERT model instance to generate embeddings
        idiom_dict (dict): Dictionary containing idioms
        save_dir (str): Directory to save the output CSV file
    
    Output:
        CSV file saved to data/embedding/CLS/[model_name].csv containing embeddings for all idioms
    """
    import pandas as pd
    from tqdm import tqdm
    import os
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get model name from instance
    model_name = model_instance.model.config._name_or_path.replace('/', '_')
    save_path = os.path.join(save_dir, f'{model_name}.csv')
    
    # Initialize empty DataFrame with column names
    columns = ['idiom'] + [f'ebd_{i}' for i in range(768)]
    all_embeddings = pd.DataFrame(columns=columns)
    
    # Process each idiom with progress bar
    for idiom in tqdm(idiom_dict.keys(), desc=f'Generating embeddings for {model_name}'):
        # Get embedding for current idiom
        embedding = model_instance.get_summarized_embedding(idiom).detach().numpy().squeeze()
        
        # Create row with idiom and embedding values
        row = [idiom] + list(embedding)
        all_embeddings.loc[len(all_embeddings)] = row
        
        # Save after each idiom in case of interruption
        all_embeddings.to_csv(save_path, index=False)

# Load idiom dictionary
with open('../data/collection/idiom.json', 'r', encoding='utf-8') as f:
    idiom_dict = json.load(f)

# Generate embeddings for each model
generate_all_embeddings(wwm, idiom_dict)
generate_all_embeddings(basic, idiom_dict)
generate_all_embeddings(wwm_large, idiom_dict)
generate_all_embeddings(mengzi, idiom_dict)

#%%
#%%




# %%
# Load final idioms data
with open('../data/final_idioms.json', 'r', encoding='utf-8') as f:
    final_idioms = json.load(f)

# Count psychological concepts
concept_counts = {}
for idiom_data in final_idioms.values():
    for concept in idiom_data['psychological_concepts']:
        concept_counts[concept] = concept_counts.get(concept, 0) + 1

# Get top 10 concepts by count
top_10_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:10]

# Organize idioms by top concepts
concept_idioms = {}
for concept, _ in top_10_concepts:
    concept_idioms[concept] = []
    for idiom, idiom_data in final_idioms.items():
        if concept in idiom_data['psychological_concepts']:
            concept_idioms[concept].append(idiom)

# Print results
for concept, idioms in concept_idioms.items():
    print(f"\n{concept}:")
    print(idioms)

# %%
# Create dictionary to store embeddings by concept
concept_embeddings = {}

# For each concept, get embeddings for each idiom
for concept, idioms in concept_idioms.items():
    concept_embeddings[concept] = {}
    
    for idiom in tqdm(idioms, desc=f'Processing concept: {concept}'):
        concept_embeddings[concept][idiom] = {}
        
        # Get embeddings from each model
        concept_embeddings[concept][idiom]['wwm'] = wwm.get_summarized_embedding(idiom)
        concept_embeddings[concept][idiom]['basic'] = basic.get_summarized_embedding(idiom)
        concept_embeddings[concept][idiom]['wwm_large'] = wwm_large.get_summarized_embedding(idiom)
        concept_embeddings[concept][idiom]['mengzi'] = mengzi.get_summarized_embedding(idiom)

# Save the concept embeddings
with open('../data/embedding/concept_embeddings.json', 'w', encoding='utf-8') as f:
    # Convert torch tensors to lists for JSON serialization
    serializable_embeddings = {}
    for concept, idioms in concept_embeddings.items():
        serializable_embeddings[concept] = {}
        for idiom, models in idioms.items():
            serializable_embeddings[concept][idiom] = {}
            for model, embedding in models.items():
                serializable_embeddings[concept][idiom][model] = embedding.tolist()
    
    json.dump(serializable_embeddings, f, ensure_ascii=False, indent=2)

# %%
def analyze_concept_similarity(concept_embeddings, model_name, sample_size):
    """Analyze similarity between idioms within each concept by sampling
    
    Args:
        concept_embeddings (dict): Dictionary containing embeddings for each concept and idiom
        model_name (str): Name of model to analyze ('wwm', 'basic', 'wwm_large', or 'mengzi')
        sample_size (int): Number of idioms to sample from each concept
        
    Returns:
        similarities (dict): Dictionary with concept keys and lists of pairwise similarities
    """
    import random
    import numpy as np
    from scipy.spatial.distance import cosine
    
    similarities = {}
    
    for concept in concept_embeddings:
        # Get list of idioms for this concept
        idioms = list(concept_embeddings[concept].keys())
        
        # Sample idioms if we have more than sample_size
        if len(idioms) > sample_size:
            sampled_idioms = random.sample(idioms, sample_size)
        else:
            sampled_idioms = idioms
            
        # Calculate pairwise similarities
        concept_sims = []
        for i in range(len(sampled_idioms)):
            for j in range(i+1, len(sampled_idioms)):
                idiom1 = sampled_idioms[i]
                idiom2 = sampled_idioms[j]
                
                # Get embeddings
                emb1 = np.array(concept_embeddings[concept][idiom1][model_name])
                emb2 = np.array(concept_embeddings[concept][idiom2][model_name])
                
                # Calculate cosine similarity
                sim = 1 - cosine(emb1, emb2)
                concept_sims.append(sim)
        
        similarities[concept] = concept_sims
        
    return similarities

analyze_concept_similarity(concept_embeddings, 'wwm', 10)
# %%
def analyze_concept_similarity(concept_idioms, model_name, sample_size):
    """Analyze similarity between idioms within each concept by sampling
    
    Args:
        concept_idioms (dict): Dictionary mapping concepts to lists of idioms
        model_name (str): Name of model to analyze ('wwm', 'basic', 'wwm_large', or 'mengzi')
        sample_size (int): Number of idioms to sample from each concept
        
    Returns:
        similarities (dict): Dictionary with concept keys and lists of pairwise similarities
    """
    import random
    import numpy as np
    from scipy.spatial.distance import cosine
    
    # Initialize model based on model_name
    if model_name == 'wwm':
        model = BertEmbedding("hfl/chinese-bert-wwm-ext")
    elif model_name == 'basic':
        model = BertEmbedding("bert-base-chinese")
    elif model_name == 'wwm_large':
        model = BertEmbedding("hfl/chinese-roberta-wwm-ext-large")
    elif model_name == 'mengzi':
        model = BertEmbedding("Langboat/mengzi-bert-base")
    
    similarities = {}
    
    for concept in tqdm(concept_idioms, desc=f'Analyzing concept similarity for {model_name}'):
        # Get list of idioms for this concept
        idioms = concept_idioms[concept]
        
        # Sample idioms if we have more than sample_size
        if len(idioms) > sample_size:
            sampled_idioms = random.sample(idioms, sample_size)
        else:
            sampled_idioms = idioms
            
        # Calculate pairwise similarities
        concept_sims = []
        for i in range(len(sampled_idioms)):
            for j in range(i+1, len(sampled_idioms)):
                idiom1 = sampled_idioms[i]
                idiom2 = sampled_idioms[j]
                
                # Get embeddings using model
                emb1 = model.get_summarized_embedding(idiom1)
                emb2 = model.get_summarized_embedding(idiom2)
                
                # Convert to numpy arrays
                emb1 = emb1.detach().numpy()
                emb2 = emb2.detach().numpy()
                
                # Calculate cosine similarity
                sim = 1 - cosine(emb1, emb2)
                concept_sims.append(sim)
        
        similarities[concept] = concept_sims
        
    return similarities

sims={}
sims['wwm']=analyze_concept_similarity(concept_idioms, 'wwm', 10)
sims['basic']=analyze_concept_similarity(concept_idioms, 'basic', 10)
sims['wwm_large']=analyze_concept_similarity(concept_idioms, 'wwm_large', 10)
sims['mengzi']=analyze_concept_similarity(concept_idioms, 'mengzi', 10)

# %%
import matplotlib.pyplot as plt
# Create lists to store data for plotting
concepts = []
wwm_sims = []
basic_sims = []

# Extract data from sims dictionary
for concept in sims['wwm'].keys():
    # Add each similarity value as a separate point
    concepts.extend([concept] * len(sims['wwm'][concept]))
    wwm_sims.extend(sims['wwm'][concept])
    basic_sims.extend(sims['basic'][concept])

# Create figure and axis
plt.figure(figsize=(12, 6))

# Plot points
plt.scatter(concepts, wwm_sims, c='#1b9e77', alpha=0.6, label='WWM')
plt.scatter(concepts, basic_sims, c='#d95f02', alpha=0.6, label='Basic')

# Customize plot
plt.xticks(rotation=45, ha='right')
plt.ylabel('Similarity', fontsize=18)
plt.xlabel('Concept', fontsize=18)
plt.title('Similarity Distribution by Concept and Model', fontsize=22)
plt.legend(fontsize=18)

# Adjust layout to prevent label cutoff
plt.tight_layout()





# %%

def analyze_concept_similarity(concept_embeddings, model_name, sample_size):
    """Analyze similarity between idioms within each concept by sampling
    
    Args:
        concept_embeddings (dict): Dictionary containing embeddings for each concept and idiom
        model_name (str): Name of model to analyze ('wwm', 'basic', 'wwm_large', or 'mengzi')
        sample_size (int): Number of idioms to sample from each concept
        
    Returns:
        similarities (dict): Dictionary with concept keys and lists of pairwise similarities
    """
    import random
    import numpy as np
    from scipy.spatial.distance import cosine
    
    similarities = {}
    
    for concept in concept_embeddings:
        # Get list of idioms for this concept
        idioms = list(concept_embeddings[concept].keys())
        
        # Sample idioms if we have more than sample_size
        if len(idioms) > sample_size:
            sampled_idioms = random.sample(idioms, sample_size)
        else:
            sampled_idioms = idioms
            
        # Calculate pairwise similarities
        concept_sims = []
        for i in range(len(sampled_idioms)):
            for j in range(i+1, len(sampled_idioms)):
                idiom1 = sampled_idioms[i]
                idiom2 = sampled_idioms[j]
                
                # Get embeddings
                emb1 = np.array(concept_embeddings[concept][idiom1][model_name])
                emb2 = np.array(concept_embeddings[concept][idiom2][model_name])
                
                # Calculate cosine similarity
                sim = 1 - cosine(emb1, emb2)
                concept_sims.append(sim)
        
        similarities[concept] = concept_sims
        
    return similarities

analyze_concept_similarity(concept_embeddings, 'wwm', 10)
# %%
def analyze_concept_similarity(concept_idioms, model_name, sample_size, seed=42):
    """Analyze similarity between idioms within each concept by sampling
    
    Args:
        concept_idioms (dict): Dictionary mapping concepts to lists of idioms
        model_name (str): Name of model to analyze ('wwm', 'basic', 'wwm_large', or 'mengzi')
        sample_size (int): Number of idioms to sample from each concept
        
    Returns:
        similarities (dict): Dictionary with concept keys and lists of pairwise similarities
    """
    import random
    import numpy as np
    from scipy.spatial.distance import cosine
    
    # Initialize model based on model_name
    if model_name == 'wwm':
        model = BertEmbedding("hfl/chinese-bert-wwm-ext")
    elif model_name == 'basic':
        model = BertEmbedding("bert-base-chinese")
    elif model_name == 'wwm_large':
        model = BertEmbedding("hfl/chinese-roberta-wwm-ext-large")
    elif model_name == 'mengzi':
        model = BertEmbedding("Langboat/mengzi-bert-base")
    
    similarities = {}
    
    for concept in tqdm(concept_idioms, desc=f'Analyzing concept similarity for {model_name}'):
        # Get list of idioms for this concept
        idioms = concept_idioms[concept]
        
        # Sample idioms if we have more than sample_size
        if len(idioms) > sample_size:
            random.seed(seed)
            sampled_idioms = random.sample(idioms, sample_size)
        else:
            sampled_idioms = idioms
            
        # Calculate pairwise similarities
        concept_sims = []
        for i in range(len(sampled_idioms)):
            for j in range(i+1, len(sampled_idioms)):
                idiom1 = sampled_idioms[i]
                idiom2 = sampled_idioms[j]
                
                # Get embeddings using model
                emb1 = model.get_summarized_embedding(idiom1)
                emb2 = model.get_summarized_embedding(idiom2)
                
                # Convert to numpy arrays
                emb1 = emb1.detach().numpy()
                emb2 = emb2.detach().numpy()
                
                # Calculate cosine similarity
                sim = 1 - cosine(emb1, emb2)
                concept_sims.append(sim)
        
        similarities[concept] = concept_sims
        
    return similarities

sims={}
sims['wwm']=analyze_concept_similarity(concept_idioms, 'wwm', 10)
sims['basic']=analyze_concept_similarity(concept_idioms, 'basic', 10)
sims['wwm_large']=analyze_concept_similarity(concept_idioms, 'wwm_large', 10)
sims['mengzi']=analyze_concept_similarity(concept_idioms, 'mengzi', 10)

# %%
import matplotlib.pyplot as plt
# Create lists to store data for plotting
concepts = []
wwm_sims = []
basic_sims = []
wwm_large_sims = []
mengzi_sims = []

# Extract data from sims dictionary
for concept in sims['wwm'].keys():
    # Add each similarity value as a separate point
    concepts.extend([concept] * len(sims['wwm'][concept]))
    wwm_sims.extend(sims['wwm'][concept])
    basic_sims.extend(sims['basic'][concept])
    wwm_large_sims.extend(sims['wwm_large'][concept])
    mengzi_sims.extend(sims['mengzi'][concept])

# Create figure and axis
plt.figure(figsize=(12, 6))

# Plot points
#plt.scatter(concepts, wwm_sims, c='#1b9e77', alpha=0.6, label='WWM')
plt.scatter(concepts, basic_sims, c='#d95f02', alpha=0.6, label='Basic')
plt.scatter(concepts, wwm_large_sims, c='#7570b3', alpha=0.6, label='WWM Large')
plt.scatter(concepts, mengzi_sims, c='#e7298a', alpha=0.6, label='Mengzi')

# Customize plot
plt.xticks(rotation=45, ha='right')
plt.ylabel('Similarity', fontsize=18)
plt.xlabel('Concept', fontsize=18)
plt.title('Similarity Distribution by Concept and Model', fontsize=22)
plt.legend(fontsize=18)

# Adjust layout to prevent label cutoff
plt.tight_layout()



# %%
