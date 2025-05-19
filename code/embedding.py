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
        
    def get_embedding(self, text, mode='avg'):
        """Get embedding vector for input text using either average pooling or CLS token
        
        Args:
            text (str): Input text to get embedding for
            mode (str): Embedding mode - 'avg' for average pooling or 'cls' for CLS token
            
        Returns:
            embedding (torch.Tensor): Embedding vector for the input text
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        
        if mode == 'avg':
            # Average all token embeddings
            embedding = outputs.last_hidden_state.mean(dim=1)
        elif mode == 'cls':
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :]
        else:
            raise ValueError("mode must be either 'avg' or 'cls'")
            
        return embedding


# mengzi
mengzi = BertEmbedding("Langboat/mengzi-bert-base")
mengzi.get_embedding("刻舟求剑")

# wwm-ext
wwm = BertEmbedding("hfl/chinese-bert-wwm-ext")
wwm.get_embedding("刻舟求剑")

# wwm-large
wwm_large = BertEmbedding("hfl/chinese-roberta-wwm-ext-large")
wwm_large.get_embedding("刻舟求剑")

# basic
basic = BertEmbedding("bert-base-chinese")
basic.get_embedding("刻舟求剑")



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
compare_benchmark(mengzi, wwm, benchmark_data)

# Compare basic and wwm-ext models
compare_benchmark(basic, wwm, benchmark_data)

# Compare wwm and wwm-large models
compare_benchmark(wwm, wwm_large, benchmark_data)

#%% embedding extraction | dataframe
def batch_embedding(instances, idioms_dict, output_file='../data/annotation/embeddings.csv'):
    """Extract and save embeddings for idioms from multiple BERT model instances into a dataframe
    
    Args:
        instances (List[BertEmbedding]): List of BERT model instances
        idioms_dict (Dict): Dictionary of idioms and their metadata
        output_file (str): Path to save embeddings CSV file
    
    Returns:
        df (pd.DataFrame): DataFrame with idioms and their embeddings
                          Columns: ['idiom', 'model', 'emb_0', 'emb_1', ..., 'emb_n']
    """
    import pandas as pd
    
    # Load existing results if any
    df = pd.DataFrame()
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        
    # Get processed idiom-model pairs and find remaining ones
    processed_pairs = set()
    if not df.empty:
        processed_pairs = set(zip(df['idiom'], df['model']))
        
    try:
        # Process each idiom with progress bar
        rows = []
        for idiom in tqdm(idioms_dict.keys(), desc="Extracting embeddings"):
            # Get embeddings from each model instance
            for instance in instances:
                model_name = instance.model.config._name_or_path
                
                # Skip if already processed
                if (idiom, model_name) in processed_pairs:
                    continue
                    
                # Get embedding and create row
                embedding = instance.get_embedding(idiom).detach().numpy().squeeze()
                row = [idiom, model_name] + embedding.tolist()
                rows.append(row)
                
                # Save intermediate results every 100 rows
                if len(rows) >= 100:
                    # Create column names if needed
                    if df.empty and rows:
                        emb_cols = [f'emb_{i}' for i in range(len(rows[0])-2)]
                        columns = ['idiom', 'model'] + emb_cols
                        df = pd.DataFrame(columns=columns)
                    
                    # Append new rows and save
                    new_df = pd.DataFrame(rows, columns=df.columns)
                    df = pd.concat([df, new_df], ignore_index=True)
                    df.to_csv(output_file, index=False)
                    rows = []
                    
        # Save any remaining rows
        if rows:
            if df.empty:
                emb_cols = [f'emb_{i}' for i in range(len(rows[0])-2)]
                columns = ['idiom', 'model'] + emb_cols
                df = pd.DataFrame(columns=columns)
            new_df = pd.DataFrame(rows, columns=df.columns)
            df = pd.concat([df, new_df], ignore_index=True)
            df.to_csv(output_file, index=False)
                
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        
    print(f"\nProcessed {len(df)} total rows")
    return df

# Load idioms dictionary
with open('../data/collection/idiom.json', 'r', encoding='utf-8') as f:
    idioms_dict = json.load(f)
    
# Extract embeddings from all models
model_instances = [basic, mengzi, wwm]
embeddings_df = batch_embedding(model_instances, idioms_dict)

# %% embedding extraction | json
def batch_embedding(instances, idioms_dict, output_file='../data/annotation/embeddings.json'):
    """Extract and save embeddings for idioms from multiple BERT model instances
    
    Args:
        instances (List[BertEmbedding]): List of BERT model instances
        idioms_dict (Dict): Dictionary of idioms and their metadata
        output_file (str): Path to save embeddings JSON file
    
    Returns:
        embeddings_dict (Dict): Dictionary mapping idioms to their embeddings from each model
                               Format: {idiom: {model_name: embedding}}
    """
    # Load existing results if any
    embeddings_dict = {}
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            embeddings_dict = json.load(f)
            
    # Get processed idioms and find remaining ones
    processed_idioms = set(embeddings_dict.keys())
    remaining_idioms = {k:v for k,v in idioms_dict.items() if k not in processed_idioms}
    
    # If no remaining idioms, return existing results
    if len(remaining_idioms) == 0:
        print(f"\nProcessed {len(embeddings_dict)} total idioms")
        print(f"Remaining unprocessed: 0")
        return embeddings_dict
        
    try:
        # Process each idiom with progress bar
        for idiom in tqdm(remaining_idioms.keys(), desc="Extracting embeddings"):
            idiom_embeddings = {}
            
            # Get embeddings from each model instance
            for instance in instances:
                model_name = instance.model.config._name_or_path
                embedding = instance.get_embedding(idiom).detach().numpy().squeeze()
                idiom_embeddings[model_name] = embedding.tolist() # Convert to list for JSON serialization
                
            # Update embeddings dict
            embeddings_dict[idiom] = idiom_embeddings
            
            # Save intermediate results
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(embeddings_dict, f, ensure_ascii=False, indent=2)
                
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        
    print(f"\nProcessed {len(embeddings_dict)} total idioms")
    print(f"Remaining unprocessed: {len(idioms_dict) - len(embeddings_dict)}")
    
    return embeddings_dict

# Load idioms dictionary
with open('../data/collection/idiom.json', 'r', encoding='utf-8') as f:
    idioms_dict = json.load(f)
    
# Extract embeddings from all models
model_instances = [basic, mengzi, wwm, wwm_large]
embeddings = batch_embedding(model_instances, idioms_dict)

#%%
# Extract embeddings from all models
model_instances = [basic, mengzi, wwm, wwm_large]
# Print embedding dimensions for each model
print("Embedding dimensions for each model:")
for instance in model_instances:
    model_name = instance.model.config._name_or_path
    sample_embedding = instance.get_embedding("测试").detach().numpy().squeeze()
    print(f"{model_name}: {len(sample_embedding)} dimensions")


#%%
from transformers import BertTokenizer, BertModel
import torch

model_name='hfl/chinese-bert-wwm-ext'
# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Define the text
text = "刻舟求剑"

# Tokenize the text
inputs = tokenizer(text, return_tensors='pt')

# Obtain the embeddings
with torch.no_grad():
    outputs = model(**inputs)

# Extract the last hidden state (embeddings)
last_hidden_states = outputs.last_hidden_state

# Print the dimensions of the embeddings
print("Shape of the last hidden state (embeddings):", last_hidden_states.shape)

# Print embeddings for each token along with their vector dimension
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
for token, embedding in zip(tokens, last_hidden_states[0]):
    print(f"Token: {token}, Embedding Dimension: {embedding.shape}, Embedding (first 5 components): {embedding[:5]}...")  # Display first 5 components for brevity
# %%
