# %% packages
import json
import pandas as pd
import random
from tqdm import tqdm

import matplotlib.pyplot as plt

import pandas as pd
import json
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

from openai import OpenAI

from tqdm import tqdm
import json
import os

import zhplot


# %% Extract idiom data
def idiom_to_dataframe(filename, limit=None):
    """
    Extract idiom data from JSON file and convert to pandas DataFrame
    
    Args:
        filename (str): Path to the JSON file containing idiom data
        limit (int, optional): Number of random idioms to sample. If None, use all idioms
    
    Returns:
        pd.DataFrame: DataFrame containing idiom words and explanations
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Optionally sample random idioms
    if limit:
        random.seed(1)
        data = random.sample(data, min(limit, len(data)))

    # Extract word and explanation pairs
    rows = [(item['word'], item['explanation']) for item in data if 'word' in item and 'explanation' in item]
    
    # Create DataFrame
    df = pd.DataFrame(rows, columns=['idiom', 'CH_meaning'])
    return df


if __name__ == "__main__":
    # the dataset I'm using here is from https://github.com/pwxcoo/chinese-xinhua, which contains 31648 idioms from Xinhua Dictionary
    df_idioms = idiom_to_dataframe("../data/definition/idiom.json", limit=None)
    print(df_idioms.head())
# %% Annotate idioms with functional categories and relevant psychological concepts
def analyze_idioms_categories(df_idioms, convert_to_df=False, gpt_model_version="gpt-4o"):
    """
    Analyze idioms and classify them into categories, get relevant psychological concepts, and English translations using LLM
    
    Input:
        df_idioms (pd.DataFrame): DataFrame containing idioms with 'idiom' and 'CH_meaning' columns
        
    Output:
        df (pd.DataFrame): DataFrame containing idiom, meaning, categories, psychological concepts, and English translation
    """
    client = OpenAI()
    
    # Convert DataFrame to dictionary
    idioms_with_meanings = dict(zip(df_idioms['idiom'], df_idioms['CH_meaning']))
    
    # Prepare prompt with all idioms
    idiom_text = "\n".join([f"{idiom}: {meaning}" for idiom, meaning in idioms_with_meanings.items()])

    prompt = f"""For each of these Chinese idioms and their meanings, please provide:

    1. Classification into these categories based on core function:
        - Mechanism Belief – Speculates on causal and structural patterns of how the world operates.
        - Behavioral Strategy – Depicts purposeful actions taken in response to specific setup.
        - Social Evaluation – Evaluates people's character, morality, or behavior.
        - Emotional Expression – Expresses emotional states or feelings.
        - Value Commitment – Claims goals, values.
        - Concrete Reference – Refers to concrete entities or states without figurative meaning.

    2. List five psychological concepts that best relate to the idiom.
    
    3. English translation of the idiom.

    Idioms:
    {idiom_text}

    Please respond in JSON format only, without any markdown code block tags:
        {{
            "idiom1": {{
                "categories": ["category1", "category2", "category3"],
                "psychological_concepts": ["concept1", "concept2", "concept3", "concept4", "concept5"],
                "english_translation": "translation"
            }},
            "idiom2": {{...}},
            ...
        }}
    """

    try:
        response = client.chat.completions.create(
            model=gpt_model_version,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        # Parse response
        result = json.loads(response.choices[0].message.content)
        
        if convert_to_df:
            # Extract data from response and construct a dataframe
            data = []
            for idiom, meaning in idioms_with_meanings.items():
                idiom_data = result.get(idiom, {})
                data.append({
                    'idiom': idiom,
                    'EN_meaning': idiom_data.get('english_translation', ''),
                    'CH_meaning': meaning,
                    'categories': idiom_data.get('categories', []),
                    'psychological_concepts': idiom_data.get('psychological_concepts', [])
                })
            return pd.DataFrame(data)
        else:
            # Convert nested dictionary to list format
            result_list = []
            for idiom, data in result.items():
                # Create new dict with idiom as first key
                new_data = {'idiom': idiom}
                # Add all other data
                new_data.update(data)
                result_list.append(new_data)
            return result_list
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(f"Raw response: {response.choices[0].message.content}")
        return None

# %% Example
test_idioms = [
    '狼心狗肺', '心灰意冷', '扬名立万', '刻舟求剑', 
    '人定胜天', '风和日丽', '勤能补拙','朽木难雕',
    '饮鸩止渴', '涸泽而渔', '一叶知秋', '敝帚自珍'
]
df_test = df_idioms[df_idioms['idiom'].isin(test_idioms)]

print(f"\nTesting {len(df_test)} specific idioms:")
result_test = analyze_idioms_categories(df_test, convert_to_df=True)

for _, row in result_test.iterrows():
    # Skip if 'Concrete Reference' is in categories
    if 'Concrete Reference' not in row['categories']:
        print(f"\nIdiom: {row['idiom']}")
        print(f"Meaning: {row['EN_meaning']}")
        print(f"Categories: {row['categories']}")
        print("Psychological Concepts:")
        for concept in row['psychological_concepts']:
            print(f"- {concept}")


# %% Process all idioms with progress bar and save results
output_file = '../data/full_idiom_analysis_results.json'
batch_size = 20

# Load existing results if any
all_results = []
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        all_results = json.load(f)
        
# Get processed idioms
processed_idioms = set(result['idiom'] for result in all_results)

# Filter unprocessed idioms
df_remaining = df_idioms[~df_idioms['idiom'].isin(processed_idioms)]

# Create batches for remaining idioms
num_batches = (len(df_remaining) + batch_size - 1) // batch_size

try:
    # Process each batch with progress bar
    for i in tqdm(range(num_batches), desc="Processing remaining idioms"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df_remaining))
        
        batch_df = df_remaining.iloc[start_idx:end_idx]
        try:
            result_list = analyze_idioms_categories(batch_df)
            all_results.extend(result_list)
                
            # Save intermediate results after each batch
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error processing batch {i}: {str(e)}")
except Exception as e:
    print(f"Fatal error in processing: {str(e)}")

print(f"\nProcessed {len(all_results)} total idioms")
print(f"Remaining unprocessed: {len(df_idioms) - len(all_results)}")
#%% output file
output_file = '../data/full_idiom_analysis_results.json'

# %% Category-based visualization
import json
import matplotlib.pyplot as plt
from collections import Counter
def visualize_category_distribution():
    """
    Visualize the distribution of categories in the idiom analysis results
    
    Input:
        None (reads from output_file)
    Output:
        None (displays plot)
    """
    # Load the JSON data
    with open(output_file, 'r', encoding='utf-8') as f:
        idiom_data = json.load(f)
    
    # Extract all categories (flatten the list since each idiom can have multiple categories)
    all_categories = []
    for item in idiom_data:
        all_categories.extend(item['categories'])
    
    # Count frequency of each category
    category_counts = Counter(all_categories)
    
    # Sort categories by frequency
    sorted_categories = dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True))
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(sorted_categories)), list(sorted_categories.values()), 
            color=plt.cm.Dark2(np.linspace(0, 1, len(sorted_categories))))
    
    # Customize plot
    plt.xticks(range(len(sorted_categories)), list(sorted_categories.keys()), 
               rotation=45, ha='right')
    plt.title('Distribution of Idiom Categories', fontsize=22)
    plt.xlabel('Category', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()

def visualize_category_overlap():
    """
    Visualize the overlap between different categories using a heatmap
    
    Input:
        None (reads from output_file)
    Output:
        None (displays heatmap)
    """
    # Load data
    with open(output_file, 'r', encoding='utf-8') as f:
        idiom_data = json.load(f)
        
    # Get unique categories
    all_categories = set()
    for item in idiom_data:
        all_categories.update(item['categories'])
    all_categories = sorted(list(all_categories))
    
    # Create overlap matrix
    n_categories = len(all_categories)
    overlap_matrix = np.zeros((n_categories, n_categories))
    
    # Fill overlap matrix
    for item in idiom_data:
        for cat1 in item['categories']:
            for cat2 in item['categories']:
                i = all_categories.index(cat1)
                j = all_categories.index(cat2)
                overlap_matrix[i,j] += 1
                
    # Create heatmap
    plt.figure(figsize=(12, 10))
    plt.imshow(overlap_matrix, cmap='YlOrRd')
    
    # Customize plot
    plt.title('Category Overlap Heatmap', fontsize=22)
    plt.xticks(range(len(all_categories)), all_categories, rotation=45, ha='right', fontsize=18)
    plt.yticks(range(len(all_categories)), all_categories, fontsize=18)
    
    # Add colorbar
    plt.colorbar()
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
def show_top_idioms_by_category(num_categories=1):
    """
    Show top 5 idioms for each category combination with specified number of categories
    
    Input:
        num_categories (int): Number of categories to consider for combinations (default: 1)
    Output:
        None (prints results)
    """
    # Load data
    with open(output_file, 'r', encoding='utf-8') as f:
        idiom_data = json.load(f)
    
    # Get unique categories
    categories = set()
    for item in idiom_data:
        categories.update(item['categories'])
    categories = sorted(list(categories))
    
    # Get all possible combinations of categories
    from itertools import combinations
    category_combinations = list(combinations(categories, num_categories))
    
    # For each combination, find matching idioms
    for category_combo in category_combinations:
        print(f"\nTop 5 idioms for combination: {' + '.join(category_combo)}")
        
        # Filter idioms with exactly these categories
        matching_idioms = [item for item in idiom_data 
                         if set(item['categories']) == set(category_combo)]
        
        # Show top 5
        random.seed(1)
        random.shuffle(matching_idioms)
        for i, idiom in enumerate(matching_idioms[:5], 1):
            print(f"{i}. {idiom['idiom']}: {idiom['EN_meaning']}")

# Generate visualizations
visualize_category_distribution()
visualize_category_overlap()
show_top_idioms_by_category(num_categories=1)
show_top_idioms_by_category(num_categories=2) 


# %% Concept-based visualization
def visualize_concept_distribution(rank_concept=None):
    """
    Visualize the distribution of psychological concepts across all idioms
    
    Input:
        rank_concept (int): 0-based index of concept to analyze, if None analyze all concepts (default: None)
    Output:
        None (displays plot)
    """
    # Load data
    with open(output_file, 'r', encoding='utf-8') as f:
        idiom_data = json.load(f)
    
    # Collect all concepts
    all_concepts = []
    for item in idiom_data:
        if rank_concept is not None:
            all_concepts.append(item['psychological_concepts'][rank_concept])
        else:
            all_concepts.extend(item['psychological_concepts'])
    
    # Count concept frequencies
    concept_counts = Counter(all_concepts)
    
    # Get top 20 concepts
    top_concepts = dict(sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:20])
    
    # Create bar plot
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(top_concepts)), list(top_concepts.values()))
    
    # Customize plot
    plt.title('Top 20 Psychological Concepts in Chinese Idioms', fontsize=22)
    plt.xlabel('Concepts', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.xticks(range(len(top_concepts)), list(top_concepts.keys()), rotation=45, ha='right', fontsize=18)
    plt.yticks(fontsize=18)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    return top_concepts

def analyze_concept_cooccurrence():
    """
    Analyze and visualize frequently co-occurring psychological concepts
    
    Input:
        None (reads from global output_file)
    Output:
        None (displays plot)
    """
    # Load data
    with open(output_file, 'r', encoding='utf-8') as f:
        idiom_data = json.load(f)
    
    # Create co-occurrence matrix
    concept_pairs = []
    for item in idiom_data:
        concepts = item['psychological_concepts']
        for i in range(len(concepts)):
            for j in range(i+1, len(concepts)):
                pair = tuple(sorted([concepts[i], concepts[j]]))
                concept_pairs.append(pair)
    
    # Count pair frequencies
    pair_counts = Counter(concept_pairs)
    
    # Get top 20 co-occurring pairs
    top_pairs = dict(sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:20])
    
    # Create bar plot
    plt.figure(figsize=(24, 8))
    plt.bar(range(len(top_pairs)), list(top_pairs.values()))
    
    # Customize plot
    plt.title('Top 20 Co-occurring Psychological Concept Pairs', fontsize=22)
    plt.xlabel('Concept Pairs', fontsize=18)
    plt.ylabel('Co-occurrence Frequency', fontsize=18)
    pair_labels = [f"{pair[0]}\n+ {pair[1]}" for pair in top_pairs.keys()]
    plt.xticks(range(len(top_pairs)), pair_labels, rotation=45, ha='right', fontsize=18)
    plt.yticks(fontsize=18)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

def show_top_idioms_by_concept(n_concepts=4, n_idioms=5):
    """
    Randomly select n_concepts psychological concepts and show n_idioms idioms for each concept
    
    Input:
        n_concepts (int): Number of concepts to randomly select (default: 4)
        n_idioms (int): Number of idioms to show per concept (default: 5)
    Output:
        None (prints results)
    """
    # Load data
    with open(output_file, 'r', encoding='utf-8') as f:
        idiom_data = json.load(f)
    
    # Get unique concepts and their frequencies
    concept_freq = {}
    for item in idiom_data:
        for concept in item['psychological_concepts']:
            concept_freq[concept] = concept_freq.get(concept, 0) + 1
    
    # Filter concepts with at least 5 instances
    valid_concepts = [c for c, freq in concept_freq.items() if freq >= 5]
    
    # Randomly select n_concepts concepts
    random.seed(33)
    selected_concepts = random.sample(valid_concepts, n_concepts)
    
    # For each selected concept, show matching idioms
    for concept in selected_concepts:
        print(f"\nIdioms containing concept: {concept}")
        
        # Filter idioms containing this concept
        matching_idioms = [item for item in idiom_data 
                         if concept in item['psychological_concepts']]
        
        # Show n_idioms random examples
        random.shuffle(matching_idioms)
        for i, idiom in enumerate(matching_idioms[:n_idioms], 1):
            print(f"{i}. {idiom['idiom']}: {idiom['EN_meaning']}")

def show_idioms_for_concepts(concepts, n_idioms=5):
    """
    Show example idioms for a given list of psychological concepts
    
    Input:
        concepts (list): List of psychological concepts to show examples for
        n_idioms (int): Number of idioms to show per concept (default: 5)
    Output:
        None (prints results)
    """
    # Load data
    with open(output_file, 'r', encoding='utf-8') as f:
        idiom_data = json.load(f)
    
    # For each concept, show matching idioms
    for concept in concepts:
        
        
        # Filter idioms containing this concept
        matching_idioms = [item for item in idiom_data 
                         if concept in item['psychological_concepts']]
        print(f"\nIdioms containing concept: {concept}, {len(matching_idioms)} idioms found")
        if not matching_idioms:
            print("No idioms found for this concept")
            continue
            
        # Show n_idioms random examples
        random.shuffle(matching_idioms)
        shown_count = 0
        i = 1
        for idiom in matching_idioms:
            if shown_count >= n_idioms:
                break
            if 'Concrete Reference' not in idiom['categories']:
                print(f"{i}. {idiom['idiom']}: {idiom['EN_meaning']}")
                #print(f"   Chinese meaning: {idiom['CH_meaning']}")
                #print(f"   Categories: {', '.join(idiom['categories'])}")
                shown_count += 1
                i += 1


# Generate visualizations
top_concepts = visualize_concept_distribution(rank_concept=0)
analyze_concept_cooccurrence()
#show_top_idioms_by_concept(n_concepts=5, n_idioms=5)
#show_idioms_for_concepts(['resilience', 'anxiety', 'communication', 'conflict', 'problem-solving','creativity'])
show_idioms_for_concepts(top_concepts.keys())

# %% fix bug
# Load the idiom data
with open('../data/full_idiom_analysis_results.json', 'r', encoding='utf-8') as f:
    idiom_data = json.load(f)

# Create mapping of idiom to meaning from original data
idiom_to_meaning = dict(zip(df_idioms['idiom'], df_idioms['CH_meaning']))

# Update each entry
for item in idiom_data:
    # Rename english_translation to EN_meaning if it exists
    if 'english_translation' in item:
        item['EN_meaning'] = item.pop('english_translation')
        
    # Add CH_meaning from original data
    if item['idiom'] in idiom_to_meaning:
        item['CH_meaning'] = idiom_to_meaning[item['idiom']]
    else:
        item['CH_meaning'] = ""  # Empty string if not found
        
# Write back to file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(idiom_data, f, ensure_ascii=False, indent=2)


# %% item examination
idiom_list = ['大器晚成','冥顽不灵']
for idiom in idiom_list:
    for item in idiom_data:
        if item['idiom'] == idiom:
            print(item)
            break

# %%
#%% structure AABB
def analyze_idiom_structure(idioms):
    """
    Analyze the character structure pattern of Chinese idioms by encoding unique characters as ABCD
    
    Input:
        idioms (list): A list of Chinese idiom strings
        
    Output:
        structure_dict (dict): Dictionary mapping idioms to their abstract structure patterns
    """
    structure_dict = {}
    
    for idiom in idioms:
        # Create mapping of unique characters to letters
        char_to_letter = {}
        structure = ''
        
        # Assign letters (A,B,C,D) to each unique character in order of appearance
        for char in idiom:
            if char not in char_to_letter:
                # Assign next available letter
                char_to_letter[char] = chr(65 + len(char_to_letter))  # 65 is ASCII for 'A'
            
            # Build structure string
            structure += char_to_letter[char]
        
        # Check if structure is just sequential letters (ABCD)
        is_sequential = True
        for i in range(len(structure)):
            if ord(structure[i]) != 65 + i:  # 65 is ASCII for 'A'
                is_sequential = False
                break
                
        structure_dict[idiom] = None if is_sequential else structure
        
    return structure_dict

# Example usage:
if __name__ == "__main__":
    # Extract idioms from idiom_data
    all_idioms = df_idioms['idiom'].tolist()
    
    # Analyze structures
    structure_patterns = analyze_idiom_structure(all_idioms)
    
    # Count frequency of each structure pattern
    pattern_counts = Counter(structure_patterns.values())
    
    # Print some examples and statistics
    print("\nMost common structure patterns:")
    for pattern, count in pattern_counts.most_common(100):
        if pattern is not None:  # Skip None patterns
            print(f"{pattern}: {count} idioms")
            # Print a few examples for each pattern
            examples = [idiom for idiom, struct in structure_patterns.items() if struct == pattern][:3]
            print(f"Examples: {', '.join(examples)}\n")

# %%
我希望对成语进行更多标注
1. 词性(复杂：主谓, 动宾; 动词 形容词
2. 倾向性： 褒义贬义中性
3. 是否包含比喻/类比
4. 结构 是否符合特定的结构：ABAB ...
5. 认知复杂度(是否包含文学典故/历史故事/语言故事)
6. descriptive/prescriptive
7. 近义词
8. 反义词

#%% metaphor
def label_metaphors(df_idioms):
    """
    Label idioms that contain explicit metaphors based on '喻' character in meanings
    
    Input:
        df_idioms (pd.DataFrame): DataFrame containing idioms with 'idiom' and 'CH_meaning' columns
        
    Output:
        pd.Series: Boolean series indicating whether each idiom contains explicit metaphor
    """
    # Create metaphor label column
    metaphor_label = df_idioms['CH_meaning'].str.contains('喻', na=False)
    return metaphor_label

# Add metaphor labels to dataframe
df_idioms['has_metaphor'] = label_metaphors(df_idioms)

# Print summary statistics
n_metaphors = df_idioms['has_metaphor'].sum()
percent = (n_metaphors / len(df_idioms)) * 100
print(f"\nFound {n_metaphors} idioms containing explicit metaphors ({percent:.2f}%)")
print("\nExample idioms with metaphorical meanings:")
print(df_idioms[df_idioms['has_metaphor']][['idiom', 'CH_meaning']].head().to_string())

# %% other annotations
def analyze_idioms_annotations(df_idioms, batch_size=10):
    """
    Analyze idioms and provide structured annotations using GPT for cognitive complexity,
    valence, syntactic structure, function type, synonyms and antonyms
    
    Input:
        df_idioms (pd.DataFrame): DataFrame containing idioms with 'idiom' and 'CH_meaning' columns
        batch_size (int): Number of idioms to process in each batch
        
    Output:
        list: List of dictionaries containing annotations for each idiom
    """
    client = OpenAI()
    
    # Convert DataFrame to dictionary
    idioms_with_meanings = dict(zip(df_idioms['idiom'], df_idioms['CH_meaning']))
    
    # Prepare prompt with all idioms
    idiom_text = "\n".join([f"{idiom}: {meaning}" for idiom, meaning in idioms_with_meanings.items()])

    prompt = f"""For each of the following Chinese idioms (with meanings), please provide the following structured annotations:

1. **Cognitive Complexity**: Can the idiom be understood solely based on its literal wording? If yes, respond with `"Literal"`. If not, specify the type of background knowledge required for understanding, choosing from: `"Religious Reference"`, `"Historical Reference"`, `"Fable or Allegory"`, or `"Literary Allusion"`.

2. **Valence**: Is the idiom predominantly positive, negative, or neutral in connotation? Choose from `"Positive"`, `"Negative"`, or `"Neutral"`.

3. **Syntactic Structure**: Classify the idiom using the following structure types:
   - `"并列结构"` (Coordinative)
   - `"偏正结构"` (Modifier-Head)
   - `"动宾结构"` (Verb-Object)
   - `"补充结构"` (Verb-Complement)
   - `"主谓结构"` (Subject-Predicate)
   - `"主谓宾结构"` (Subject-Verb-Object)
   - `"连谓结构"` (Serial-Verb)
   - `"兼语结构"` (Double-Object or Pivot)
   - `"复指结构"` (Coreference)
   - `"介宾结构"` (Preposition-Object)
   - `"固定结构"` (Fixed Expression)

4. **Prescriptive or Descriptive**: 
- Prescriptive: the idiom primarily offer guidance, advice, or behavioral norms
- Descriptive: the idiom simply describe a state, condition, or observed pattern without offering any guidance or advice

5. **Synonyms**: List as many **idioms** as possible that are semantically or functionally similar.

6. **Antonyms**: List as many **idioms** as possible that convey opposite meanings or intentions.

Idioms:
{idiom_text}

Please respond in JSON format only, **without any markdown code block tags**:
{{
    "idiom1": {{
        "idiom": idiom1,
        "cognitive_complexity": str (Literal, Religious Reference, Historical Reference, Fable or Allegory, Literary Allusion),
        "valence": str (Positive, Negative, Neutral), 
        "syntactic_structure": str (并列结构, 偏正结构, 动宾结构, 补充结构, 主谓结构, 主谓宾结构, 连谓结构, 兼语结构, 复指结构, 介宾结构, 固定结构),
        "function_type": str (Descriptive, Prescriptive),
        "synonyms": list,
        "antonyms": list
    }},
    ...
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2000
        )
        
        # Parse response
        # Remove markdown code block tags if present
        content = response.choices[0].message.content
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        result = json.loads(content)
        
        # Convert nested dictionary to list format
        result_list = []
        for idiom, data in result.items():
            # Create new dict with idiom as first key
            new_data = {'idiom': idiom}
            # Add all other data
            new_data.update(data)
            result_list.append(new_data)
        return result_list
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(f"Raw response: {response.choices[0].message.content}")
        return None

analyze_idioms_annotations(df_idioms[:20])

# %% Process all idioms with progress bar and save results
output_file = '../data/additional_idiom_annotations.json'
error_file = '../data/failed_idioms.json'
batch_size = 20

# Load existing results if any
all_annotations = []
failed_idioms = []
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        all_annotations = json.load(f)

if os.path.exists(error_file):
    with open(error_file, 'r', encoding='utf-8') as f:
        failed_idioms = json.load(f)
        
# Get processed idioms
processed_idioms = set(result['idiom'] for result in all_annotations)
failed_idiom_set = set(item['idiom'] for item in failed_idioms)

# Filter unprocessed idioms
df_remaining = df_idioms[~df_idioms['idiom'].isin(processed_idioms | failed_idiom_set)]

# Create batches for remaining idioms
num_batches = (len(df_remaining) + batch_size - 1) // batch_size

try:
    # Process each batch with progress bar
    for i in tqdm(range(num_batches), desc="Processing remaining idioms"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df_remaining))
        
        batch_df = df_remaining.iloc[start_idx:end_idx]
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                result_list = analyze_idioms_annotations(batch_df)
                if result_list:
                    all_annotations.extend(result_list)
                    # Save intermediate results after each batch
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(all_annotations, f, ensure_ascii=False, indent=2)
                    break
                retry_count += 1
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    print(f"Error processing batch {i} after {max_retries} attempts: {str(e)}")
                    # Save failed idioms
                    for _, row in batch_df.iterrows():
                        failed_idioms.append({
                            'idiom': row['idiom'],
                            'error': str(e),
                            'batch': i
                        })
                    with open(error_file, 'w', encoding='utf-8') as f:
                        json.dump(failed_idioms, f, ensure_ascii=False, indent=2)
                else:
                    print(f"Retry {retry_count} for batch {i}")
                    time.sleep(1)  # Wait before retrying
                    
except Exception as e:
    print(f"Fatal error in processing: {str(e)}")

print(f"\nProcessed {len(all_annotations)} total idioms")
print(f"Failed {len(failed_idioms)} idioms")
print(f"Remaining unprocessed: {len(df_idioms) - len(all_annotations) - len(failed_idioms)}")
# %%
# %% Analyze distributions of different features
import pandas as pd
import json

# Load additional idiom annotations
with open('../data/additional_idiom_annotations.json', 'r', encoding='utf-8') as f:
    additional_data = json.load(f)

# Convert to DataFrame
df_additional = pd.DataFrame(additional_data)

# Calculate distributions
print("\nCognitive Complexity Distribution:")
print(df_additional['cognitive_complexity'].value_counts(normalize=True))

print("\nValence Distribution:")
print(df_additional['valence'].value_counts(normalize=True))

print("\nSyntactic Structure Distribution:")
print(df_additional['syntactic_structure'].value_counts(normalize=True))

print("\nFunction Type Distribution:")
print(df_additional['function_type'].value_counts(normalize=True))

# %% Merge analysis results
# Load full idiom analysis
with open('../data/full_idiom_analysis_results.json', 'r', encoding='utf-8') as f:
    full_analysis = json.load(f)

# Convert to DataFrame
df_full = pd.DataFrame(full_analysis)

# Load additional idiom analysis
with open('../data/additional_idiom_annotations.json', 'r', encoding='utf-8') as f:
    additional_analysis = json.load(f)

# Convert to DataFrame
df_additional_analysis = pd.DataFrame(additional_analysis)

# Merge on idiom field
df_merged = pd.merge(df_full, df_additional_analysis, on='idiom', how='outer')


#%% interaction plot
def analyze_feature_by_concept(df, feature_col, concept_type='psychological_concepts', top_n=20):
    """
    Analyze distribution of features for each concept/domain
    
    Input:｜
        df (DataFrame): Merged dataframe containing idiom analysis
        feature_col (str): Column name of the feature to analyze
        concept_type (str): Either 'psychological_concepts' or 'domains'
        top_n (int): Number of top concepts to display
    
    Output:
        None (displays plot)
    """
    # Explode the concepts list
    df_exploded = df.explode(concept_type)
    
    # Get top N concepts by frequency
    top_concepts = df_exploded[concept_type].value_counts().nlargest(top_n).index
    
    # Filter for top concepts
    df_filtered = df_exploded[df_exploded[concept_type].isin(top_concepts)]
    
    # Calculate proportions
    props = pd.crosstab(df_filtered[concept_type], 
                       df_filtered[feature_col], 
                       normalize='index')
    
    # Sort by the first level's proportion
    first_level = props.columns[0]
    props = props.sort_values(by=first_level, ascending=False)
    
    # Create stacked bar plot
    plt.figure(figsize=(25, 10))  # Increased figure size
    props.plot(kind='bar', stacked=True)
    #plt.title(f'Distribution of {feature_col} by {concept_type}', fontsize=22)
    plt.xlabel(concept_type.replace('_', ' ').title(), fontsize=18)
    plt.ylabel('Proportion', fontsize=18)
    plt.xticks(rotation=45, ha='right', fontsize=18)
    plt.yticks(fontsize=18)
    
    # Calculate number of columns for legend based on number of categories
    ncol = min(4, len(props.columns))  # Maximum 4 columns, adjust if needed
    plt.legend(fontsize=14, bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=ncol)
    
    plt.tight_layout()
    plt.show()

# Analyze features by psychological concepts
for feature in ['valence', 'cognitive_complexity', 'syntactic_structure', 'function_type']:
    analyze_feature_by_concept(df_merged, feature, 'psychological_concepts')
    
# Analyze features by domains
for feature in ['valence', 'cognitive_complexity', 'syntactic_structure', 'function_type']:
    analyze_feature_by_concept(df_merged, feature, 'domains')

# %%
#替换字

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from textwrap import wrap

def analyze_feature_by_concept(df, feature_col, concept_type='psychological_concepts', top_n=20):
    df_exploded = df.explode(concept_type)
    top_concepts = df_exploded[concept_type].value_counts().nlargest(top_n).index
    df_filtered = df_exploded[df_exploded[concept_type].isin(top_concepts)]

    props = pd.crosstab(df_filtered[concept_type], 
                        df_filtered[feature_col], 
                        normalize='index')
    first_level = props.columns[0]
    props = props.sort_values(by=first_level, ascending=False)

    # 自定义颜色（可修改）
    palette = sns.color_palette("Set2", n_colors=len(props.columns))

    fig, ax = plt.subplots(figsize=(14, 6))
    props.plot(kind='bar', stacked=True, ax=ax, color=palette)

    # 美化
    ax.set_xlabel(concept_type.replace('_', ' ').title(), fontsize=20)
    ax.set_ylabel('Proportion', fontsize=20)
    ax.tick_params(axis='x', labelsize=16, rotation=45)
    ax.tick_params(axis='y', labelsize=16)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')

    # 自动缩短X轴文字
    new_labels = ['\n'.join(wrap(l, 20)) for l in props.index]
    ax.set_xticklabels(new_labels)

    # 图例美化
    ax.legend(title=feature_col.replace('_', ' ').title(),
              fontsize=14, title_fontsize=16,
              bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)

    # 去掉顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


# Analyze features by psychological concepts
for feature in ['valence', 'cognitive_complexity', 'syntactic_structure', 'function_type']:
    analyze_feature_by_concept(df_merged, feature, 'psychological_concepts')
    
# Analyze features by domains
for feature in ['valence', 'cognitive_complexity', 'syntactic_structure', 'function_type']:
    analyze_feature_by_concept(df_merged, feature, 'categories')
# %%


import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from textwrap import wrap

def analyze_feature_by_concept(df, feature_col, concept_type='psychological_concepts', top_n=20):
    """
    Analyze distribution of features for each concept/domain using heatmap visualization
    
    Input:
        df (DataFrame): Merged dataframe containing idiom analysis
        feature_col (str): Column name of the feature to analyze
        concept_type (str): Either 'psychological_concepts' or 'domains'
        top_n (int): Number of top concepts to display
    
    Output:
        None (displays plot)
    """
    # Prepare data
    df_exploded = df.explode(concept_type)
    top_concepts = df_exploded[concept_type].value_counts().nlargest(top_n).index
    df_filtered = df_exploded[df_exploded[concept_type].isin(top_concepts)]

    # Calculate proportions
    props = pd.crosstab(df_filtered[concept_type], 
                       df_filtered[feature_col], 
                       normalize='index')
    
    # Sort by the first level's proportion
    first_level = props.columns[0]
    props = props.sort_values(by=first_level, ascending=False)

    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(props, 
                cmap='YlOrRd',
                annot=True, 
                fmt='.2f',
                cbar_kws={'label': 'Proportion'},
                annot_kws={'size': 10})

    # Customize labels and title
    plt.xlabel(feature_col.replace('_', ' ').title(), fontsize=18)
    plt.ylabel(concept_type.replace('_', ' ').title(), fontsize=18)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right', fontsize=18)
    plt.yticks(rotation=0, fontsize=18)
    
    # Wrap y-axis labels
    ylabels = ['\n'.join(wrap(l, 20)) for l in props.index]
    plt.gca().set_yticklabels(ylabels)

    plt.tight_layout()
    plt.show()


# Analyze features by psychological concepts
for feature in ['valence', 'cognitive_complexity', 'syntactic_structure', 'function_type']:
    analyze_feature_by_concept(df_merged, feature, 'psychological_concepts')
    
# Analyze features by domains
for feature in ['valence', 'cognitive_complexity', 'syntactic_structure', 'function_type']:
    analyze_feature_by_concept(df_merged, feature, 'categories')
# %%
#1) 将cognitive complexity里的除了history fable literary literal都改成literal 2) 改列名 3）

#%%
# Data cleaning and standardization
# Fix Sentiment values
df_merged['valence'] = df_merged['valence'].replace('Prescriptive', 'Positive')

# Standardize Background Knowledge values
background_keep = ['Historical Reference', 'Fable or Allegory', 'Literary Allusion', 'Literal']
df_merged['cognitive_complexity'] = df_merged['cognitive_complexity'].apply(
    lambda x: x if x in background_keep else 'Literal'
)

# Convert Chinese syntactic structure terms to English
structure_mapping = {
    '并列结构': 'Coordinative',
    '偏正结构': 'Modifier-Head', 
    '动宾结构': 'Verb-Object',
    '补充结构': 'Verb-Complement',
    '主谓结构': 'Subject-Predicate',
    '主谓宾结构': 'Subject-Verb-Object',
    '连谓结构': 'Serial-Verb',
    '兼语结构': 'Double-Object',
    '复指结构': 'Coreference',
    '介宾结构': 'Preposition-Object',
    '固定结构': 'Fixed Expression'
}
df_merged['syntactic_structure'] = df_merged['syntactic_structure'].map(structure_mapping)

# Rename columns
df_merged = df_merged.rename(columns={
    'cognitive_complexity': 'background_knowledge',
    'valence': 'sentiment'
})

# Analyze features by psychological concepts
for feature in ['sentiment', 'background_knowledge', 'syntactic_structure', 'function_type']:
    analyze_feature_by_concept(df_merged, feature, 'psychological_concepts')
    
# Analyze features by domains
for feature in ['sentiment', 'background_knowledge', 'syntactic_structure', 'function_type']:
    analyze_feature_by_concept(df_merged, feature, 'categories')
# %%
def find_idioms_by_concept(json_file, field, target, n=5):
    """Find idioms that contain a specific target concept in their categories or psychological concepts
    
    Args:
        json_file (str): Path to the JSON file containing idiom analysis
        field (str): Field to search in ('categories' or 'psychological_concepts')
        target (str): Target concept to search for
        n (int): Number of idioms to return (default 5)
    
    Returns:
        list: List of dictionaries containing idiom info that match the target concept
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    matching_idioms = []
    for item in data:
        random.shuffle(data)
        if field in item and target in item[field]:
            matching_idioms.append({
                'idiom': item['idiom'],
                'EN_meaning': item['EN_meaning'],
                'CH_meaning': item['CH_meaning'],
                field: item[field]
            })
            if len(matching_idioms) >= n:
                break
                
    return matching_idioms

# Example usage:
idioms = find_idioms_by_concept('../data/full_idiom_analysis_results.json', 'psychological_concepts', 'Wisdom', 5)
print(f"Idioms related to Wisdom: {idioms}")


# %%
print(find_idioms_by_concept('../data/full_idiom_analysis_results.json', 'categories', 'Concrete Reference', 5))
print(find_idioms_by_concept('../data/full_idiom_analysis_results.json', 'categories', 'Value Commitment', 5))
print(find_idioms_by_concept('../data/full_idiom_analysis_results.json', 'categories', 'Emotional Expression', 5))
print(find_idioms_by_concept('../data/full_idiom_analysis_results.json', 'categories', 'Social Evaluation', 5))
print(find_idioms_by_concept('../data/full_idiom_analysis_results.json', 'categories', 'Mechanism Belief', 5))
print(find_idioms_by_concept('../data/full_idiom_analysis_results.json', 'categories', 'Behavioral Strategy', 5))

# %%
def analyze_semantic_relationships(df_merged, json_file, field):
    """Analyze semantic relationships (synonyms/antonyms) within categories/concepts
    
    Args:
        df_merged (DataFrame): DataFrame containing idiom relationships
        json_file (str): Path to JSON file with idiom analysis
        field (str): Field to analyze ('categories' or 'psychological_concepts')
        
    Returns:
        dict: Dictionary with counts of synonyms and antonyms per category/concept
    """
    # Load idiom analysis data
    with open(json_file, 'r', encoding='utf-8') as f:
        idiom_data = json.load(f)
    
    # Create lookup dictionary for idiom categories/concepts
    idiom_lookup = {}
    for item in idiom_data:
        if field in item and len(item[field]) > 0:
            # Only take the first category/concept
            idiom_lookup[item['idiom']] = item[field][0]
    
    # Initialize counter dictionary
    relationship_counts = {}
    
    # Analyze each idiom's relationships
    for idx, row in df_merged.iterrows():
        base_idiom = row['idiom']
        if base_idiom not in idiom_lookup:
            continue
            
        # Get base idiom's category/concept
        base_category = idiom_lookup[base_idiom]

        # Check synonyms
        if isinstance(row['synonyms'], list):
            synonyms = row['synonyms']
            for syn in synonyms:
                if syn in idiom_lookup:
                    syn_category = idiom_lookup[syn]
                    # Count if categories match
                    if base_category == syn_category:
                        if base_category not in relationship_counts:
                            relationship_counts[base_category] = {'synonyms': 0, 'antonyms': 0}
                        relationship_counts[base_category]['synonyms'] += 1
        # Check antonyms
        if isinstance(row['antonyms'], list):
            antonyms = row['antonyms']
            for ant in antonyms:
                if ant in idiom_lookup:
                    ant_category = idiom_lookup[ant]
                    # Count if categories match
                    if base_category == ant_category:
                        if base_category not in relationship_counts:
                            relationship_counts[base_category] = {'synonyms': 0, 'antonyms': 0}
                        relationship_counts[base_category]['antonyms'] += 1
    
    return relationship_counts

# Analyze semantic relationships for categories and psychological concepts
category_relationships = analyze_semantic_relationships(df_merged, '../data/full_idiom_analysis_results.json', 'categories')
concept_relationships = analyze_semantic_relationships(df_merged, '../data/full_idiom_analysis_results.json', 'psychological_concepts')

print("\nCategory Semantic Relationships:")
for cat, counts in category_relationships.items():
    print(f"{cat}: Synonyms={counts['synonyms']}, Antonyms={counts['antonyms']}")

print("\nPsychological Concept Semantic Relationships:")
for concept, counts in concept_relationships.items():
    print(f"{concept}: Synonyms={counts['synonyms']}, Antonyms={counts['antonyms']}")

# %%
def plot_semantic_relationships(relationships, title):
    """
    Plot semantic relationships (synonyms/antonyms/Not_Identified) distribution for categories/concepts
    
    Input:
        relationships (dict): Dictionary containing synonym/antonym counts per category
        title (str): Title for the plot
    
    Output:
        None (displays plot)
    """
    # Convert dictionary to DataFrame
    data = []
    for category, counts in relationships.items():
        data.append({
            'Category': category,
            'Synonyms': counts['synonyms'],
            'Antonyms': counts['antonyms'],
            'Not_Identified': counts['Not_Identified']
        })
    df = pd.DataFrame(data)
    
    # Calculate proportions
    total = df[['Synonyms', 'Antonyms', 'Not_Identified']].sum(axis=1)
    df['Synonyms_prop'] = df['Synonyms'] / total
    df['Antonyms_prop'] = df['Antonyms'] / total
    df['Not_Identified_prop'] = df['Not_Identified'] / total
    
    # Sort by total relationships and take top 20
    df['Total'] = total
    df = df.sort_values('Total', ascending=False).head(20)
    
    # Create stacked bar plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Use Dark2 colormap for discrete data
    colors = plt.cm.Dark2([0, 1, 2])
    # Sort by synonym proportion
    df = df.sort_values('Synonyms_prop', ascending=False)
    
    df[['Synonyms_prop', 'Antonyms_prop', 'Not_Identified_prop']].plot(
        kind='bar', 
        stacked=True, 
        ax=ax,
        color=colors
    )
    
    # Customize plot
    ax.set_xlabel('Categories', fontsize=18)
    ax.set_ylabel('Proportion', fontsize=18)
    ax.tick_params(axis='x', labelsize=18, rotation=45)
    ax.tick_params(axis='y', labelsize=18)
    
    # Wrap x-axis labels
    new_labels = ['\n'.join(wrap(l, 20)) for l in df['Category']]
    ax.set_xticklabels(new_labels, ha='right')
    
    # Customize legend
    ax.legend(['Synonyms', 'Antonyms', 'No Relations'], 
             fontsize=18,
             bbox_to_anchor=(1.05, 1), 
             loc='upper left')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    # Add title based on the field type
    if 'Category' in df.columns:
        plt.title('Semantic Consistency by Category', fontsize=22)
    else:
        plt.title('Semantic Consistency by Psychological Concept', fontsize=22)
    plt.show()

def analyze_semantic_relationships_all_concepts(df_merged, json_file, field):
    """
    Analyze semantic relationships considering all concepts for each idiom
    
    Input:
        df_merged (DataFrame): DataFrame containing idiom relationships
        json_file (str): Path to JSON file with idiom analysis
        field (str): Field to analyze ('categories' or 'psychological_concepts')
    
    Output:
        dict: Dictionary with counts of synonyms, antonyms and no relations per category/concept
    """
    # Load idiom analysis data
    with open(json_file, 'r', encoding='utf-8') as f:
        idiom_data = json.load(f)
    
    # Create lookup dictionary for idiom categories/concepts
    idiom_lookup = {}
    for item in idiom_data:
        if field in item and len(item[field]) > 0:
            idiom_lookup[item['idiom']] = item[field]  # Store all concepts
    
    # Initialize counter dictionary
    relationship_counts = {}
    
    # Analyze each idiom's relationships
    for idx, row in df_merged.iterrows():
        base_idiom = row['idiom']
        if base_idiom not in idiom_lookup:
            continue
            
        # Get all base idiom's categories/concepts
        base_categories = idiom_lookup[base_idiom]
        
        # Initialize has_relation flag for each category
        category_has_relation = {cat: False for cat in base_categories}

        # Check synonyms
        if isinstance(row['synonyms'], list):
            for syn in row['synonyms']:
                if syn in idiom_lookup:
                    syn_categories = idiom_lookup[syn]
                    # Count shared categories
                    shared_categories = set(base_categories) & set(syn_categories)
                    for category in shared_categories:
                        if category not in relationship_counts:
                            relationship_counts[category] = {'synonyms': 0, 'antonyms': 0, 'Not_Identified': 0}
                        relationship_counts[category]['synonyms'] += 1
                        category_has_relation[category] = True

        # Check antonyms
        if isinstance(row['antonyms'], list):
            for ant in row['antonyms']:
                if ant in idiom_lookup:
                    ant_categories = idiom_lookup[ant]
                    # Count shared categories
                    shared_categories = set(base_categories) & set(ant_categories)
                    for category in shared_categories:
                        if category not in relationship_counts:
                            relationship_counts[category] = {'synonyms': 0, 'antonyms': 0, 'Not_Identified': 0}
                        relationship_counts[category]['antonyms'] += 1
                        category_has_relation[category] = True
        
        # Count idioms with no relations for each category
        for category, has_relation in category_has_relation.items():
            if not has_relation:
                if category not in relationship_counts:
                    relationship_counts[category] = {'synonyms': 0, 'antonyms': 0, 'Not_Identified': 0}
                relationship_counts[category]['Not_Identified'] += 1
    
    return relationship_counts

# Analyze semantic relationships for categories and psychological concepts using all concepts
category_relationships = analyze_semantic_relationships_all_concepts(df_merged, '../data/full_idiom_analysis_results.json', 'categories')
concept_relationships = analyze_semantic_relationships_all_concepts(df_merged, '../data/full_idiom_analysis_results.json', 'psychological_concepts')

# Plot relationships for both categories and concepts
plot_semantic_relationships(category_relationships, "Category Semantic Relationships")
plot_semantic_relationships(concept_relationships, "Psychological Concept Semantic Relationships")

# %% Add character template and metaphor features
# Get character template patterns
all_idioms = df_merged['idiom'].tolist()
structure_patterns = analyze_idiom_structure(all_idioms)

# Get top 6 non-sequential patterns
pattern_counts = Counter([p for p in structure_patterns.values() if p is not None])
top_patterns = dict(pattern_counts.most_common(6))

# Add character template patterns to df_merged, only keeping top 6 non-sequential
df_merged['character_template'] = df_merged['idiom'].map(
    lambda x: structure_patterns[x] if structure_patterns[x] in top_patterns else None
)

# Add metaphor labels
df_merged['metaphor'] = label_metaphors(df_merged)

# Print metaphor statistics
n_metaphors = df_merged['metaphor'].sum()
percent = (n_metaphors / len(df_merged)) * 100
print(f"\nFound {n_metaphors} idioms containing explicit metaphors ({percent:.2f}%)")
print("\nExample idioms with metaphorical meanings:")
print(df_merged[df_merged['metaphor']][['idiom', 'CH_meaning']].head().to_string())

# Plot heatmaps for both features, filtering out None values for character_template
df_plot = df_merged[df_merged['character_template'].notna()]
analyze_feature_by_concept(df_plot, 'character_template', 'psychological_concepts')
analyze_feature_by_concept(df_merged, 'metaphor', 'psychological_concepts')
analyze_feature_by_concept(df_plot, 'character_template', 'categories')
analyze_feature_by_concept(df_merged, 'metaphor', 'categories')
# %%
# %% Process remaining idioms and update annotations
output_file = '../data/additional_idiom_annotations.json'

# Load existing annotations
all_annotations = []
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        all_annotations = json.load(f)

# Get processed idioms
processed_idioms = set(result['idiom'] for result in all_annotations)

# Find unprocessed idioms
df_remaining = df_idioms[~df_idioms['idiom'].isin(processed_idioms)]
print(f"\nFound {len(df_remaining)} unprocessed idioms")

# Process remaining idioms in batches
batch_size = 10
num_batches = (len(df_remaining) + batch_size - 1) // batch_size

try:
    # Process each batch with progress bar
    for i in tqdm(range(num_batches), desc="Processing remaining idioms"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df_remaining))
        
        batch_df = df_remaining.iloc[start_idx:end_idx]
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                result_list = analyze_idioms_annotations(batch_df)
                if result_list:
                    all_annotations.extend(result_list)
                    # Save intermediate results after each batch
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(all_annotations, f, ensure_ascii=False, indent=2)
                    break
                retry_count += 1
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    print(f"Error processing batch {i} after {max_retries} attempts: {str(e)}")
                    # Split batch in half and try each half
                    mid = len(batch_df) // 2
                    first_half = batch_df.iloc[:mid]
                    second_half = batch_df.iloc[mid:]
                    
                    print("Trying first half of failed batch...")
                    try:
                        result_list = analyze_idioms_annotations(first_half)
                        if result_list:
                            all_annotations.extend(result_list)
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(all_annotations, f, ensure_ascii=False, indent=2)
                    except Exception as e1:
                        print(f"Error processing first half: {str(e1)}")
                        
                    print("Trying second half of failed batch...")
                    try:
                        result_list = analyze_idioms_annotations(second_half)
                        if result_list:
                            all_annotations.extend(result_list)
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(all_annotations, f, ensure_ascii=False, indent=2)
                    except Exception as e2:
                        print(f"Error processing second half: {str(e2)}")
                else:
                    print(f"Retry {retry_count} for batch {i}")
                    time.sleep(1)  # Wait before retrying
                    
except Exception as e:
    print(f"Fatal error in processing: {str(e)}")

print(f"\nProcessed {len(all_annotations)} total idioms")
print(f"Remaining unprocessed: {len(df_idioms) - len(all_annotations)}")
# %%





# %%
import pandas as pd
df_idioms = pd.read_json("../data/collection/idiom.json", orient='records', 
lines=True)
print(f"The idiom dataset has {len(df_idioms)} idioms")
print(df_idioms.head())

# %%
# Convert to dictionary with idiom as key
idioms_dict = {}
for _, row in df_idioms.iterrows():
    idiom = row['idiom']
    idioms_dict[idiom] = {
        'CH_meaning': row['CH_meaning'],
        'derivation': row['derivation']
    }

# Save as JSON file
with open('../data/collection/idiom.json', 'w', encoding='utf-8') as f:
    json.dump(idioms_dict, f, ensure_ascii=False, indent=2)

# %% convert to dict
# Convert list[dict] to dict{dict} format and reorder fields
def convert_list_to_dict(input_path, output_path, field_order=None, field_mapping=None):
    """
    Convert annotation file from list[dict] to dict{dict} format and reorder fields
    
    Input:
        input_path (str): Path to input annotation file
        output_path (str): Path to save converted annotation file
        field_order (list): List of field names in desired order
        field_mapping (dict): Mapping of old field names to new field names
    Output:
        dict: Converted annotations
    """
    # Load input file
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to dict format
    converted = {}
    for item in data:
        idiom = item.pop('idiom')
        
        # Apply field mapping if provided
        if field_mapping:
            mapped_item = {}
            for old_name, value in item.items():
                new_name = field_mapping.get(old_name, old_name)
                mapped_item[new_name] = value
            item = mapped_item
            
        if field_order:
            # Reorder fields if order specified
            ordered_item = {field: item[field] for field in field_order if field in item}
            converted[idiom] = ordered_item
        else:
            converted[idiom] = item
            
    # Save converted annotations
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
        
    return converted

# Convert concept mapping annotations
concept_field_order = ['categories', 'psychological_concepts','EN_meaning'
]

convert_list_to_dict(
    '../data/concept_mapping.json',
    '../data/annotation/concept_mapping.json',
    field_order=concept_field_order
)

# Convert structural analysis annotations 
structural_field_order = [
    'background_knowledge', 'sentiment', 'syntactic_structure',
    'function_type', 'synonyms', 'antonyms'
]

structural_field_mapping = {
    'cognitive_complexity': 'background_knowledge',
    'valence': 'sentiment'
}

convert_list_to_dict(
    '../data/structural_annotation.json', 
    '../data/annotation/structural_analysis.json',
    field_order=structural_field_order,
    field_mapping=structural_field_mapping）



# 筛选4字



# %%
# Load concept mapping and idioms files
with open('../data/annotation/concept_mapping.json', 'r', encoding='utf-8') as f:
    concept_mapping = json.load(f)
with open('../data/collection/idiom.json', 'r', encoding='utf-8') as f:
    idioms = json.load(f)

# Get sets of idioms from each file
concept_idioms = set(concept_mapping.keys())
all_idioms = set(idioms.keys())

# Find idioms unique to each file
only_in_concept = concept_idioms - all_idioms
only_in_idioms = all_idioms - concept_idioms

print("\nIdioms only in concept_mapping.json:")
print(f"Total count: {len(only_in_concept)}")
for idiom in sorted(only_in_concept):
    print(f"- {idiom}")

print("\nIdioms only in idiom.json:")
print(f"Total count: {len(only_in_idioms)}")
for idiom in sorted(only_in_idioms):
    print(f"- {idiom}")

# %%
with open('../data/annotation/concept_mapping.json', 'r', encoding='utf-8') as f:
    concept_mapping = json.load(f)
with open('../data/collection/idiom.json', 'r', encoding='utf-8') as f:
    idioms = json.load(f)
    
# Remove extra idioms
extra_idioms = set(concept_mapping.keys()) - set(idioms.keys())
for idiom in extra_idioms:
    del concept_mapping[idiom]
with open('../data/annotation/concept_mapping.json', 'w', encoding='utf-8') as f:
    json.dump(concept_mapping, f, ensure_ascii=False, indent=2)
    
# Add missing idioms through batch annotation
batch_annotation(idioms, '../data/annotation/concept_mapping.json', annotate_concept_mapping)
        

