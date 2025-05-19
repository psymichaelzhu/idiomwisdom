# %% [markdown]
# Objective: 
# 1. Given a dataset of idioms (json file), containing the following fields:
#    - CH_meaning: the Chinese meaning of the idiom
#    - derivation: the derivation of the idiom
# 2. Concept Mapping: Annotate the dataset with the following fields: 
#    - categories: the functional categories of the idiom (e.g. "social evaluation")
#    - psychological_concepts: the psychological concepts of the idiom (e.g. "growth mindset")
#    - EN_meaning: the English meaning of the idiom 
# 3. Structure Analysis: Annotate the dataset with the following fields: 
#    - background_knowledge: whether the idiom requires background knowledge to understand
#    - sentiment: the sentiment of the idiom
#    - syntactic_structure: the syntactic structure of the idiom
#    - function_type: whether the idiom is prescriptive or descriptive
#    - synonyms: the synonyms of the idiom
#    - antonyms: the antonyms of the idiom
# 4. Rule-based Analysis: The following two structural fields are derived from rule-based analysis instead of LLM: 
#    - has_metaphor: whether the idiom employs metaphor/analogy
#    - character_template: the special character arrangement pattern of the idiom (e.g. AABB)
# 5. Save the annotated dataset as a json file


# %% [markdown]
# notes. 
# - Currently, we are using GPT-4o to annotate the dataset, later we might have human experts and rule-based methods to annotate the dataset in a fine-grained way.

# - all the datasets (both input and output) have idiom as the key, and the value is a dictionary containing the fields. For example, {"勤能补拙": {'CH_meaning': '勤奋能够弥补不足。', 'derivation': '宋·邵雍《弄笔吟》弄假像真终是假，将勤补拙总轮勤。”'}}

# - *usage*: 
#   - for each annotation code block: 
#     - change exmaple False to True to run the example
#     - change batch False to True to run all idioms by batch 
#   - after you get all the annotation files, you can merge them into a single file by running the merge_annotations code block
#   - This will give you a full annotated dataset (input + all annotations)

# %% packages
import json
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

# %% Load the dataset
with open("../data/collection/idiom.json", "r", encoding="utf-8") as f:
    idioms_dict = json.load(f)
print(f"The idiom dataset has {len(idioms_dict)} idioms")
print(idioms_dict['勤能补拙']) #example

# %% helper functions
def parse_gpt_response(response_content):
    """
    Parse GPT response by removing markdown code block tags and converting to JSON (just in case)
    
    Input:
        response_content (str): Raw response content from GPT
    Output:
        result (dict): Parsed JSON result
    """
    # Remove markdown code block tags if present
    content = response_content.strip()
    if content.startswith('```json'):
        content = content[7:]
    if content.endswith('```'):
        content = content[:-3]
    content = content.strip()
    result = json.loads(content)
    return result

def annotation_examples(idiom_list, idioms_dict, annotation_func, print_result=True):
    """
    Analyze a list of example idioms using the provided annotation function and print results
    
    Input:
        idiom_list (list): List of idioms to analyze
        idioms_dict (dict): Dictionary containing idiom data
        annotation_func (function): Function to annotate idioms
        print_result (bool): Whether to print the results
        
    Output:
        dict: Dictionary containing idiom annotation results (if print_result is False)
    """
    # Get subset of idioms
    subset_dict = {idiom: idioms_dict[idiom] for idiom in idiom_list if idiom in idioms_dict}
    
    # Run annotation
    print(f"\nAnalyzing {len(subset_dict)} idioms:")
    results = annotation_func(subset_dict)
    
    # Print results
    if print_result:
        for idiom, data in results.items():
            print(f"\nIdiom: {idiom}")
            for key, value in data.items():
                print(f"{key}: {value}")
    else:
        return results 

def batch_annotation(input_dict, output_file, annotation_func, batch_size=20):
    """
    Process idioms in batches using an annotation function and save results, with progress bar, error catching, intermediate saving, and retry mechanism
    
    Input:
        input_dict (dict): Dictionary containing idioms to process
        output_file (str): Path to JSON file to save results
        annotation_func (callable): Function that takes dictionary and returns dict of annotation dicts
        batch_size (int): Size of batches to process at once, if None process all at once
    
    Output:
        all_results (dict): Dictionary of all processed annotation dictionaries
    """
    # Load existing results if any
    all_results = {}
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
            
    # Get processed idioms and find remaining ones
    processed_idioms = set(all_results.keys())
    remaining_dict = {k:v for k,v in input_dict.items() if k not in processed_idioms}
    
    # If no remaining idioms, return all results, don't process
    if len(remaining_dict) == 0:
        print(f"\nProcessed {len(all_results)} total idioms")
        print(f"Remaining unprocessed: 0")
        return all_results
    
    if batch_size is None:
        batch_size = len(remaining_dict)
        
    # Create batches from dictionary
    idiom_items = list(remaining_dict.items())
    
    num_batches = (len(idiom_items) + batch_size - 1) // batch_size
    
    try:
        # Process each batch with progress bar
        for i in tqdm(range(num_batches), desc="Processing idioms"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(idiom_items))
            batch_dict = dict(idiom_items[start_idx:end_idx])
            
            # Try processing with decreasing batch sizes
            current_batch_size = batch_size
            max_retries = 3
            success = False
            
            for attempt in range(max_retries):
                try:
                    result_dict = annotation_func(batch_dict)
                    
                    # Update existing results
                    all_results.update(result_dict)
                            
                    # Save intermediate results
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(all_results, f, ensure_ascii=False, indent=2)
                        
                    success = True
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        current_batch_size = current_batch_size // 2
                        print(f"Retry {attempt + 1} with batch size {current_batch_size}")
                        batch_dict = dict(list(batch_dict.items())[:current_batch_size])
                    else:
                        print(f"Error processing batch {i} after {max_retries} attempts: {str(e)}")
                        
            if not success:
                print(f"Skipping problematic batch starting at index {start_idx}")
                
    except Exception as e:
        print(f"Fatal error in processing: {str(e)}")
        
    print(f"\nProcessed {len(all_results)} total idioms")
    print(f"Remaining unprocessed: {len(input_dict) - len(all_results)}")
    
    return all_results


# %% Concept Mapping
def annotate_concept_mapping(idioms_dict, gpt_model_version="gpt-4o"):
    """
    Concept Mapping: 1) Translate idioms into English, 2) Classify idioms into categories, 3) Get relevant psychological concepts using LLM
    
    Input:
        idioms_dict (dict): Dictionary containing idioms where keys are idioms and values are dictionaries with 'CH_meaning' field
        gpt_model_version (str): The GPT model version to use
        
    Output:
        dict: A dictionary where keys are idioms and values are dictionaries containing:
            - categories (list[str]): List of functional categories
            - psychological_concepts (list[str]): List of relevant psychological concepts
            - EN_meaning (str): The English translation of the idiom
    """
    
    client = OpenAI()
    
    # Create text input from idioms and their meanings
    idiom_text = "\n".join(f"{idiom}: {data['CH_meaning']}" for idiom, data in idioms_dict.items())

    prompt = f"""For each of these Chinese idioms (with meanings), please provide:

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
                "EN_meaning": "translation"
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
            temperature=0 #for reproducibility
        )
        
        # Parse response and return directly since it's already in the desired format
        return parse_gpt_response(response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(f"Raw response: {response.choices[0].message.content}")
        return None
# example
if False: 
   annotation_examples([
    '狼心狗肺', '心灰意冷', '扬名立万', '刻舟求剑', 
    '人定胜天', '风和日丽', '勤能补拙','朽木难雕',
    '饮鸩止渴', '涸泽而渔', '一叶知秋', '敝帚自珍'
    ], 
    idioms_dict, annotate_concept_mapping)
# batch
if False: 
    batch_annotation(idioms_dict, "../data/annotation/concept_mapping.json", annotate_concept_mapping)

# %% Structural Analysis
def annotate_structural_analysis(idioms_dict, gpt_model_version="gpt-4o"):
    """
    Structural Analysis: 1) Analyze background knowledge, 2) Determine sentiment, 3) Identify syntactic structure, 4) Classify function type (prescriptive or descriptive), 5) Find synonyms and antonyms
    
    Input:
        idioms_dict (dict): Dictionary containing idioms where keys are idioms and values are dictionaries with 'CH_meaning' field
        gpt_model_version (str): The GPT model version to use for analysis
        
    Output:
        dict: A dictionary where keys are idioms and values are dictionaries containing:
            - background_knowledge (str): Type of knowledge required to understand the idiom
            - sentiment (str): Positive, negative or neutral sentiment
            - syntactic_structure (str): Grammatical structure type
            - function_type (str): Whether prescriptive or descriptive
            - synonyms (list[str]): List of semantically similar idioms
            - antonyms (list[str]): List of idioms with opposite meanings
    """
    client = OpenAI()
    
    # Create text input from idioms and their meanings
    idiom_text = "\n".join(f"{idiom}: {data['CH_meaning']}" for idiom, data in idioms_dict.items())

    prompt = f"""For each of the following Chinese idioms (with meanings), please provide the following structured annotations:

1. **Background Knowledge**: Can the idiom be understood solely based on its literal wording? If yes, respond with `"Literal"`. If not, specify the type of background knowledge required for understanding, choosing from: `"Religious Reference"`, `"Historical Reference"`, `"Fable or Allegory"`, or `"Literary Allusion"`.

2. **Sentiment**: Is the idiom predominantly positive, negative, or neutral in connotation? Choose from `"Positive"`, `"Negative"`, or `"Neutral"`.

3. **Syntactic Structure**: Classify the idiom using the following structure types:
   - `"Coordinative"` (并列结构)
   - `"Modifier-Head"` (偏正结构)
   - `"Verb-Object"` (动宾结构)
   - `"Verb-Complement"` (补充结构)
   - `"Subject-Predicate"` (主谓结构)
   - `"Subject-Verb-Object"` (主谓宾结构)
   - `"Serial-Verb"` (连谓结构)
   - `"Double-Object or Pivot"` (兼语结构)
   - `"Coreference"` (复指结构)
   - `"Preposition-Object"` (介宾结构)
   - `"Fixed Expression"` (固定结构)

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
        "background_knowledge": str (Literal, Religious Reference, Historical Reference, Fable or Allegory, Literary Allusion),
        "sentiment": str (Positive, Negative, Neutral), 
        "syntactic_structure": str (Coordinative, Modifier-Head, Verb-Object, Verb-Complement, Subject-Predicate, Subject-Verb-Object, Serial-Verb, Double-Object or Pivot, Coreference, Preposition-Object, Fixed Expression),
        "function_type": str (Descriptive, Prescriptive),
        "synonyms": list,
        "antonyms": list
    }},
    ...
}}"""

    try:
        response = client.chat.completions.create(
            model=gpt_model_version,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0 #for reproducibility
        )
        
        # Parse response and return directly since it's already in the desired format
        return parse_gpt_response(response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(f"Raw response: {response.choices[0].message.content}")
        return None
# example
if False: 
    annotation_examples([
    '狼心狗肺', '心灰意冷', '扬名立万', '刻舟求剑', 
    '人定胜天', '风和日丽', '勤能补拙','朽木难雕',
    '饮鸩止渴', '涸泽而渔', '一叶知秋', '敝帚自珍'
    ], 
    idioms_dict, annotate_structural_analysis)
# batch
if False: 
    batch_annotation(idioms_dict, "../data/annotation/structural_analysis.json", annotate_structural_analysis)

#%% Rule-based Analysis
def annotate_rulebased_analysis(idioms_dict):
    """
    Analyze idioms using rule-based methods including 1) metaphor/analogy detection and 2) character arrangement patterns
    
    Input:
        idioms_dict (dict): Dictionary containing idioms where keys are idioms and values are dictionaries with 'CH_meaning' field
        
    Output:
        dict: Dictionary where keys are idioms and values are dictionaries containing:
            - has_metaphor (bool): Whether idiom contains explicit metaphor/analogy
            - character_template (str): The special arrangement of characters in the idiom (e.g. AABB)
    """
    results = {}
    
    # Analyze each idiom
    for idiom, data in idioms_dict.items():
        CH_meaning = data['CH_meaning']
        
        # Check for metaphor
        # if the idiom CH_meaning contains '喻' (chinese metaphor indicator), it is a metaphor/analogy; this is a relatively strict criterion
        has_metaphor = '喻' in CH_meaning if CH_meaning else False
        
        # Analyze structure pattern
        char_to_letter = {}
        structure = ''
        for char in idiom:
            if char not in char_to_letter:
                char_to_letter[char] = chr(65 + len(char_to_letter))
            structure += char_to_letter[char]
            
        # Check if sequential pattern (e.g. ABCD), which means no special arrangement of characters
        is_sequential = True
        for i in range(len(structure)):
            if ord(structure[i]) != 65 + i:
                is_sequential = False
                break
        
        character_template = None if is_sequential else structure
        
        # Add results
        results[idiom] = {
            'has_metaphor': has_metaphor,
            'character_template': character_template
        }
        
    return results
# example
if False: 
    annotation_examples([
    '守株待兔', '自怨自艾'
    ], 
    idioms_dict, annotate_rulebased_analysis)
# batch
if False: 
    batch_annotation(idioms_dict, "../data/annotation/rulebased_analysis.json", annotate_rulebased_analysis, batch_size=None)

# %% Merge annotations
def merge_annotations(annotation_paths, input_path, output_path):
    """
    Merge multiple annotation files and input file into a single file (taking the intersection of idioms)
    
    Input:
        annotation_paths (list[str]): List of paths to annotation files
        input_path (str): Path to input idiom file containing original data
        output_path (str): Path to save the merged annotations
    Output:
        dict: Dictionary containing merged annotations for each idiom
    """
    # Combine all paths
    all_paths = annotation_paths + [input_path]
    
    # Load all files
    annotation_dicts = [
        json.load(open(f, 'r', encoding='utf-8'))
        for f in all_paths
    ]
    
    # Print number of idioms in each file
    for path, annot_dict in zip(all_paths, annotation_dicts):
        print(f"Number of idioms in {path.split('/')[-1]}: {len(annot_dict)}")
    
    # Get intersection of idioms across all dictionaries
    all_idioms = set.intersection(*[set(d.keys()) for d in annotation_dicts])
    
    # Merge annotations
    merged_annotations = {}
    for idiom in all_idioms:
        merged_annotations[idiom] = {}
        for annot_dict in annotation_dicts:
            merged_annotations[idiom].update(annot_dict[idiom])
            
    # Print number of idioms after merging
    print(f"\nNumber of idioms after merging: {len(merged_annotations)}")
    
    # Save merged annotations
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_annotations, f, ensure_ascii=False, indent=2)
        
    return merged_annotations

# Example usage
if True:
    merged_annotations = merge_annotations(annotation_paths=[
        '../data/annotation/rulebased_analysis.json',
        '../data/annotation/concept_mapping.json',
        '../data/annotation/structural_analysis.json'],
        input_path='../data/collection/idiom.json',
        output_path='../data/final_idioms.json')

