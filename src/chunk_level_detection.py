import sys
sys.path.insert(0, '/content/drive/MyDrive/transformers/src')  # Updated path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer 
import json
from torch.nn import functional as F
from tqdm import tqdm
import pdb
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Script for processing data and models.')
parser.add_argument('--model_name', type=str, required=True, help='llama2-7b or llama2-13b or llama3-8b')
parser.add_argument(
    '--dataset', 
    type=str, 
    default="ragtruth", 
    help='ragtruth, dolly, or my_dataset'
)
# Add custom dataset path arguments
parser.add_argument('--custom_response_path', type=str, default=None, help='Path to custom response file')
parser.add_argument('--custom_source_path', type=str, default=None, help='Path to custom source info file')

args = parser.parse_args()

# Base directory - Update this to your Drive folder structure
BASE_DIR = "/content/drive/MyDrive/ReDeEP-ICLR"  # Updated to match your path  # Updated to match your path

bge_model = SentenceTransformer('BAAI/bge-base-en-v1.5').to("cuda:0")

# Dataset paths
if args.dataset == "ragtruth":
    if args.model_name == "llama3-8b":
        response_path = f"{BASE_DIR}/dataset/ragtruth/response_span_with_llama3_8b.jsonl"
    else:
        response_path = f"{BASE_DIR}/dataset/ragtruth/response_spans.jsonl"
    source_info_path = f"{BASE_DIR}/dataset/ragtruth/source_info_spans.jsonl"
elif args.dataset == "dolly":
    response_path = f"{BASE_DIR}/dataset/ragtruth/response_dolly_spans.jsonl"
    source_info_path = f"{BASE_DIR}/dataset/ragtruth/source_info_dolly_spans.jsonl"
elif args.dataset == "my_dataset":
    # Use custom paths or default paths
    if args.custom_response_path:
        response_path = args.custom_response_path
    else:
        response_path = f"{BASE_DIR}/dataset/ragtruth/my_dataset_response_spans.jsonl"
    
    if args.custom_source_path:
        source_info_path = args.custom_source_path
    else:
        source_info_path = f"{BASE_DIR}/dataset/ragtruth/my_dataset_source_info_spans.jsonl"
else:
    print(f"Unknown dataset: {args.dataset}")
    print("Please use: ragtruth, dolly, or my_dataset")
    exit(-1)

# Check if files exist
if not os.path.exists(response_path):
    print(f"Error: Response file not found: {response_path}")
    exit(-1)
if not os.path.exists(source_info_path):
    print(f"Error: Source info file not found: {source_info_path}")
    exit(-1)

response = []
with open(response_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        response.append(data)

source_info_dict = {}
with open(source_info_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        source_info_dict[data['source_id']] = data

# Model paths - Update these to your Drive location
MODEL_BASE_DIR = f"{BASE_DIR}/models"  # or wherever you stored your models

if args.model_name == "llama2-7b":
    model_name = "llama-2-7b-chat-hf"
    model_path = f"{MODEL_BASE_DIR}/llama2/{model_name}"
elif args.model_name == "llama2-13b":
    model_name = "llama-2-13b-chat-hf"
    model_path = f"{MODEL_BASE_DIR}/llama2/{model_name}"
elif args.model_name == "llama3-8b":
    model_name = "Meta-Llama-3-8B-Instruct"
    model_path = f"{MODEL_BASE_DIR}/llama3/{model_name}"
else:
    print("name error")
    exit(-1)

# Check if model path exists
if not os.path.exists(model_path):
    print(f"Error: Model path does not exist: {model_path}")
    print("Please update MODEL_BASE_DIR or download the model to the correct location")
    exit(-1)

print(f"Loading model from: {model_path}")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = "cuda"

if args.model_name == "llama2-13b":
    tokenizer_for_temp = AutoTokenizer.from_pretrained(f"{MODEL_BASE_DIR}/llama2/llama-2-7b-chat-hf")
else:
    tokenizer_for_temp = tokenizer

# Top-k heads paths
if args.model_name == "llama2-7b":
    topk_head_path = f"{BASE_DIR}/log/test_llama2_7B/topk_heads.json"
elif args.model_name == "llama2-13b":
    topk_head_path = f"{BASE_DIR}/log/test_llama2_13B/topk_heads.json"
elif args.model_name == "llama3-8b":
    topk_head_path = f"{BASE_DIR}/log/test_llama3_8B/topk_heads.json"
else:
    print("model name error")
    exit(-1)

if not os.path.exists(topk_head_path):
    print(f"Error: Top-k heads file not found: {topk_head_path}")
    exit(-1)

with open(topk_head_path,'r') as f:
    copy_heads = json.load(f)[:32]


def calculate_dist(sep_vocabulary_dist, sep_attention_dist):
    softmax_mature_layer = F.softmax(sep_vocabulary_dist, dim=-1)  
    softmax_anchor_layer = F.softmax(sep_attention_dist, dim=-1)  

    M = 0.5 * (softmax_mature_layer + softmax_anchor_layer) 

    log_softmax_mature_layer = F.log_softmax(sep_vocabulary_dist, dim=-1)  
    log_softmax_anchor_layer = F.log_softmax(sep_attention_dist, dim=-1) 

    kl1 = F.kl_div(log_softmax_mature_layer, M, reduction='none').mean(-1)  
    kl2 = F.kl_div(log_softmax_anchor_layer, M, reduction='none').mean(-1)  
    js_divs = 0.5 * (kl1 + kl2) 
        
    return js_divs.cpu().item()*10e5


def calculate_dist_2d(sep_vocabulary_dist, sep_attention_dist):
    softmax_mature_layer = F.softmax(sep_vocabulary_dist, dim=-1)  
    softmax_anchor_layer = F.softmax(sep_attention_dist, dim=-1)  

    M = 0.5 * (softmax_mature_layer + softmax_anchor_layer) 

    log_softmax_mature_layer = F.log_softmax(sep_vocabulary_dist, dim=-1)  
    log_softmax_anchor_layer = F.log_softmax(sep_attention_dist, dim=-1)   

    kl1 = F.kl_div(log_softmax_mature_layer, M, reduction='none').sum(dim=-1)  
    kl2 = F.kl_div(log_softmax_anchor_layer, M, reduction='none').sum(dim=-1)  
    js_divs = 0.5 * (kl1 + kl2)
    
    scores = js_divs.cpu().tolist()
    
    return sum(scores)

def calculate_ma_dist(sep_vocabulary_dist, sep_attention_dist):
    sep_vocabulary_dist = F.softmax(sep_vocabulary_dist, dim=-1)

    dist_diff = sep_vocabulary_dist - sep_attention_dist
    abs_diff = torch.abs(dist_diff)
    manhattan_distance = torch.sum(abs_diff)
    
    return manhattan_distance.cpu().item()

def add_special_template(prompt):
    messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
    text = tokenizer_for_temp.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text

def is_hallucination_token(token_id, hallucination_spans):
    for span in hallucination_spans:
        if token_id >= span[0] and token_id <= span[1]:
            return True
    return False

def is_hallucination_span(r_span, hallucination_spans):
    for token_id in range(r_span[0], r_span[1]):
        for span in hallucination_spans:
            if token_id >= span[0] and token_id <= span[1]:
                return True
    return False

def calculate_hallucination_spans(response, text, response_rag, tokenizer, prefix_len):
    hallucination_span = []
    if "dolly" in source_info_path or args.dataset == "my_dataset":
        return hallucination_span
    for item in response:
        start_id = item['start']
        end_id = item['end']
        start_text = text+response_rag[:start_id]
        end_text = text+response_rag[:end_id]
        start_text_id = tokenizer(start_text, return_tensors="pt").input_ids
        end_text_id = tokenizer(end_text, return_tensors="pt").input_ids
        start_id = start_text_id.shape[-1]
        end_id = end_text_id.shape[-1]
        hallucination_span.append([start_id, end_id])
    return hallucination_span

def calculate_respond_spans(raw_response_spans, text, response_rag, tokenizer):
    respond_spans = []
    for item in raw_response_spans:
        start_id = item[0]
        end_id = item[1]
        start_text = text+response_rag[:start_id]
        end_text = text+response_rag[:end_id]
        start_text_id = tokenizer(start_text, return_tensors="pt").input_ids
        end_text_id = tokenizer(end_text, return_tensors="pt").input_ids
        start_id = start_text_id.shape[-1]
        end_id = end_text_id.shape[-1]
        respond_spans.append([start_id, end_id])
    return respond_spans


def calculate_prompt_spans(raw_prompt_spans, prompt, tokenizer):
    prompt_spans = []
    for item in raw_prompt_spans:
        start_id = item[0]
        end_id = item[1]
        start_text = prompt[:start_id]
        end_text = prompt[:end_id]
        added_start_text = add_special_template(start_text)
        added_end_text = add_special_template(end_text)
        start_text_id = tokenizer(added_start_text, return_tensors="pt").input_ids.shape[-1] - 4
        end_text_id = tokenizer(added_end_text,return_tensors="pt").input_ids.shape[-1] -4
        prompt_spans.append([start_text_id, end_text_id])
    return prompt_spans

def calculate_sentence_similarity(r_text, p_text):
    part_embedding = bge_model.encode([r_text], normalize_embeddings=True)
    q_embeddings = bge_model.encode([p_text], normalize_embeddings=True)
    scores_named = np.matmul(q_embeddings, part_embedding.T).flatten()
    return float(scores_named[0])


select_response = []
if args.model_name == "llama2-7b":
    data_type = "llama-2-7b-chat"
elif args.model_name == "llama2-13b":
    data_type = "llama-2-13b-chat"
elif args.model_name == "llama3-8b":
    data_type =  "llama-3-8b-instruct" 
else:
    print("model name error")
    exit(-1) 

print(f"\nProcessing {len(response)} responses...")
processed_count = 0

for i in tqdm(range(len(response))):
    # For custom dataset, skip model type check if not present
    if args.dataset == "my_dataset":
        if "split" in response[i] and response[i]["split"] != "test":
            continue
    else:
        if response[i]['model'] != data_type or response[i]["split"] != "test":
            continue
    
    processed_count += 1
    response_rag = response[i]['response']
    source_id = response[i]['source_id']
    temperature = response[i].get('temperature', 1.0)
    prompt =  source_info_dict[source_id]['prompt']
    original_prompt_spans = source_info_dict[source_id]['prompt_spans']
    original_response_spans = response[i]['response_spans']

    text = add_special_template(prompt[:12000])
    input_text = text+response_rag
    print(f"\nProcessing sample {processed_count}:")
    print("all_text_len:", len(input_text))
    print("prompt_len", len(prompt))
    print("respond_len", len(response_rag))
    
    input_ids = tokenizer([input_text], return_tensors="pt").input_ids
    prefix_ids = tokenizer([text], return_tensors="pt").input_ids
    continue_ids = input_ids[0, prefix_ids.shape[-1]:]

    if "labels" in response[i].keys():
        hallucination_spans = calculate_hallucination_spans(response[i]['labels'], text, response_rag, tokenizer, prefix_ids.shape[-1])
    else:
        hallucination_spans = []

    prompt_spans = calculate_prompt_spans(source_info_dict[source_id]['prompt_spans'], prompt, tokenizer)
    respond_spans = calculate_respond_spans(response[i]['response_spans'], text, response_rag, tokenizer)
    
    if args.model_name == "llama2-7b":
        start = 0 
        number = 32
    elif args.model_name == "llama3-8b":
        start = 0
        number = 16
    elif args.model_name == "llama2-13b":
        start = 8
        number = 40
    else:
        print("model name error")

    start_p, end_p = None, None
    with torch.no_grad():
      outputs = model(
        input_ids=input_ids,
        return_dict=True,
        output_attentions=True,
        output_hidden_states=True,
        knowledge_layers=list(range(start, number))
    )

# if you had a custom logits_dict before, reconstruct it manually
# here, assume you want to use outputs.logits as base
    logits = outputs.logits  # shape: [batch, seq_len, vocab_size]
    logits_dict = {"model_logits": [logits.to(device), logits.to(device)]}


    hidden_states = outputs["hidden_states"]
    last_hidden_states = hidden_states[-1][0, :, :]
    
    external_similarity = []
    parameter_knowledge_difference = []
    hallucination_label = []
    
    span_socre_dict = []
    for r_id, r_span in enumerate(respond_spans):
        layer_head_span = {}
        for attentions_layer_id in range(len(outputs.attentions)):
            for head_id in range(outputs.attentions[attentions_layer_id].shape[1]):
                if [attentions_layer_id, head_id] in copy_heads:
                    layer_head = (attentions_layer_id, head_id)
                    p_span_score_dict = []
                    for p_span in prompt_spans:
                        attention_score = outputs.attentions[attentions_layer_id][0,head_id,:,:]
                        p_span_score_dict.append([p_span, torch.sum(attention_score[r_span[0]:r_span[1], p_span[0]:p_span[1]]).cpu().item()])
                    
                    p_id = max(range(len(p_span_score_dict)), key=lambda i: p_span_score_dict[i][1])
                    prompt_span_text = prompt[original_prompt_spans[p_id][0]:original_prompt_spans[p_id][1]]
                    respond_span_text = response_rag[original_response_spans[r_id][0]:original_response_spans[r_id][1]]
                    
                    layer_head_span[str(layer_head)] = calculate_sentence_similarity(prompt_span_text, respond_span_text)

        parameter_knowledge_scores = [calculate_dist_2d(value[0][0,r_span[0]:r_span[1],:], value[1][0,r_span[0]:r_span[1],:]) for value in logits_dict.values()]
        parameter_knowledge_dict = {f"layer_{i}": value for i, value in enumerate(parameter_knowledge_scores)}

        span_socre_dict.append({
            "prompt_attention_score":layer_head_span,
            "r_span": r_span,
            "hallucination_label": 1 if is_hallucination_span(r_span, hallucination_spans) else 0,
            "parameter_knowledge_scores": parameter_knowledge_dict
        }) 

    response[i]["scores"] = span_socre_dict
    select_response.append(response[i])

print(f"\nProcessed {processed_count} samples")

# Save paths - create directories if they don't exist
if args.model_name == "llama2-7b":
    log_dir = f"{BASE_DIR}/log/test_llama2_7B"
    if args.dataset == "ragtruth":
        save_path = f"{log_dir}/llama2_7B_response_chunk.json"
    elif args.dataset == "dolly":
        save_path = f"{log_dir}/llama2_7B_response_chunk_dolly.json"
    elif args.dataset == "my_dataset":
        save_path = f"{log_dir}/llama2_7B_response_chunk_my_dataset.json"
elif args.model_name == "llama2-13b":
    log_dir = f"{BASE_DIR}/log/test_llama2_13B"
    if args.dataset == "ragtruth":
        save_path = f"{log_dir}/llama2_13B_response_chunk.json"
    elif args.dataset == "dolly":
        save_path = f"{log_dir}/llama2_13B_response_chunk_dolly.json"
    elif args.dataset == "my_dataset":
        save_path = f"{log_dir}/llama2_13B_response_chunk_my_dataset.json"
elif args.model_name == "llama3-8b":
    log_dir = f"{BASE_DIR}/log/test_llama3_8B"
    if args.dataset == "ragtruth":
        save_path = f"{log_dir}/llama3_8B_response_chunk.json"
    elif args.dataset == "dolly":
        save_path = f"{log_dir}/llama3_8B_response_chunk_dolly.json"
    elif args.dataset == "my_dataset":
        save_path = f"{log_dir}/llama3_8B_response_chunk_my_dataset.json"
else:
    print("model name error")
    exit(-1)

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(save_path), exist_ok=True)

print(f"\nSaving results to: {save_path}")
with open(save_path, "w") as f:
    json.dump(select_response, f, ensure_ascii=False)

print("Done!")