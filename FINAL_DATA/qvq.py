import time
import torch
import json
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import re
import os

# Dataset file mappings
dataset_files = {
    'AGI_EVAL': 'AGI_EVAL.json',
    'EMMA_MATH': 'emma_math_dataset.json',
    'EMMA_PHYSICS': 'emma_physics_dataset.json',
    'GPQA': 'gpqa_physics_dataset.json',
    'GSM8K': 'gsm8k_dataset.json',
    'JEEBENCH': 'jeebench_dataset.json',
    'MATHVISTA': 'mathvista_dataset.json',
    'MMLU_MATH': 'mmlu_math_physics_test.json',
    'MMMU_ARCH': 'mmmu_arch_eng_test_dataset.json',
    'OLYMPIAD': 'olympiad_bench_dataset.json',
    'SCIBENCH': 'scibench_dataset.json',
    'SCIENCEQA': 'scienceqa_test_dataset.json',
    'SCIEVAL_PHYSICS': 'scieval_physics_combined.json'
}

# Load model
model_id = "Qwen/QVQ-72B-Preview"

print("Loading tokenizer and processor...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

# Fix pad token issue and set left padding for decoder-only models
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Set left padding for decoder-only models
tokenizer.padding_side = 'left'

print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)

print("Model loaded successfully!")

# Initialize results dictionary
results = {}

# JSON formatting instruction
instruction = """
Please analyze this question and provide your response in the following JSON format:
{
  "answer": "Your answer here (numerical value or option letter for MCQ, or null for proof/open-ended questions)",
  "solution": "Detailed step-by-step solution"
}
Return ONLY valid JSON without any additional text.
"""

# Batch processing configuration
BATCH_SIZE = 4  # Adjust based on your GPU memory

def has_images(question_obj):
    """Check if question has images"""
    if 'image' in question_obj:
        if isinstance(question_obj['image'], list) and len(question_obj['image']) > 0:
            return True
        elif isinstance(question_obj['image'], str) and question_obj['image']:
            return True
    return False

def extract_json_from_response(response):
    """Extract JSON from model response"""
    try:
        # First, try direct JSON parsing
        parsed_response = json.loads(response)
        return parsed_response
    except json.JSONDecodeError:
        # If that fails, try to extract JSON using regex pattern
        json_pattern = r'({[\s\S]*?})'
        match = re.search(json_pattern, response)
        if match:
            try:
                parsed_response = json.loads(match.group(1))
                return parsed_response
            except json.JSONDecodeError:
                pass
        
        # If extraction fails, store the raw response
        return {"error": "Failed to parse response as JSON", "raw_response": response}

def create_qvq_message(question_text, images=None):
    """Create QvQ message format"""
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."}
            ],
        }
    ]
    
    # Create user content
    user_content = []
    
    # Add images if present
    if images:
        for img_path in images:
            user_content.append({"type": "image", "image": img_path})
    
    # Add text
    user_content.append({"type": "text", "text": f"{question_text}\n\n{instruction}"})
    
    messages.append({
        "role": "user",
        "content": user_content
    })
    
    return messages

def process_text_batch(questions_batch):
    """Process batch of text-only questions"""
    if not questions_batch:
        return []
        
    messages_batch = []
    
    for question_text in questions_batch:
        messages = create_qvq_message(question_text)
        messages_batch.append(messages)
    
    # Prepare texts for batch processing
    texts = []
    for messages in messages_batch:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)
    
    # Process batch
    inputs = processor(
        text=texts,
        images=None,  # Explicitly set to None for text-only
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    # Generate - removed invalid parameters
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=8192,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False
    )
    
    # Decode only the new tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    responses = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    
    return responses

def process_multimodal_batch(questions_batch):
    """Process batch of multimodal questions"""
    if not questions_batch:
        return []
        
    messages_batch = []
    all_image_inputs = []
    
    for question_text, images in questions_batch:
        messages = create_qvq_message(question_text, images)
        messages_batch.append(messages)
        
        # Extract images for this question
        image_inputs, _ = process_vision_info(messages)  # We ignore video_inputs
        if image_inputs:
            all_image_inputs.extend(image_inputs)
    
    # Prepare texts for batch processing
    texts = []
    for messages in messages_batch:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)
    
    # Process batch
    inputs = processor(
        text=texts,
        images=all_image_inputs if all_image_inputs else None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    # Generate - removed invalid parameters
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=8192,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False
    )
    
    # Decode only the new tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    responses = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    
    return responses

def print_sample_results(results, dataset_name, num_samples=3):
    """Print sample results for debugging and monitoring"""
    if not results:
        return
        
    print(f"\n{'='*30} SAMPLE RESULTS for {dataset_name} {'='*30}")
    
    # Get a few sample results
    sample_results = results[:num_samples] if len(results) >= num_samples else results
    
    for i, result in enumerate(sample_results):
        print(f"\n--- Sample {i+1} ---")
        print(f"ID: {result['id']}")
        print(f"Has Images: {result['has_images']}")
        print(f"Question: {result['question'][:100]}..." if len(result['question']) > 100 else f"Question: {result['question']}")
        
        if isinstance(result['response'], dict):
            if 'error' in result['response']:
                print(f"Response: ERROR - {result['response']['error']}")
            else:
                print(f"Answer: {result['response'].get('answer', 'N/A')}")
                solution = result['response'].get('solution', 'N/A')
                print(f"Solution: {solution[:200]}..." if len(str(solution)) > 200 else f"Solution: {solution}")
        else:
            print(f"Response: {str(result['response'])[:200]}...")
    
    print(f"{'='*80}")

def process_questions_in_batches(questions, dataset_name):
    """Process questions in batches, separating text and multimodal"""
    text_questions = []
    multimodal_questions = []
    question_indices = []
    
    # Separate questions by type
    for idx, question_obj in enumerate(questions):
        question_text = question_obj.get('question', '')
        
        if has_images(question_obj):
            images = question_obj['image']
            if isinstance(images, str):
                images = [images]
            multimodal_questions.append((question_text, images))
        else:
            text_questions.append(question_text)
        
        question_indices.append((idx, has_images(question_obj)))
    
    all_responses = [None] * len(questions)
    processed_results = []
    
    # Process text questions in batches
    if text_questions:
        print(f"Processing {len(text_questions)} text questions in batches of {BATCH_SIZE}")
        text_idx = 0
        
        for i in tqdm(range(0, len(text_questions), BATCH_SIZE), desc=f"Text batches for {dataset_name}"):
            batch = text_questions[i:i+BATCH_SIZE]
            
            # Skip empty batches
            if not batch:
                continue
                
            try:
                batch_responses = process_text_batch(batch)
                
                # Map responses back to original indices
                for response in batch_responses:
                    # Find the corresponding original index
                    while text_idx < len(question_indices) and question_indices[text_idx][1]:  # Skip multimodal
                        text_idx += 1
                    if text_idx < len(question_indices):
                        all_responses[question_indices[text_idx][0]] = response
                        
                        # Create result entry for sample printing
                        orig_idx = question_indices[text_idx][0]
                        question_obj = questions[orig_idx]
                        parsed_response = extract_json_from_response(response)
                        
                        result_entry = {
                            "id": question_obj.get('id', orig_idx),
                            "question": question_obj.get('question', ''),
                            "response": parsed_response,
                            "has_images": False
                        }
                        processed_results.append(result_entry)
                        text_idx += 1
                        
            except Exception as e:
                print(f"Error processing text batch {i//BATCH_SIZE + 1}: {str(e)}")
                # Fill with error responses
                for j in range(len(batch)):
                    while text_idx < len(question_indices) and question_indices[text_idx][1]:
                        text_idx += 1
                    if text_idx < len(question_indices):
                        all_responses[question_indices[text_idx][0]] = f"Batch processing error: {str(e)}"
                        text_idx += 1
            
            # Print sample results every 10 batches
            if (i//BATCH_SIZE + 1) % 10 == 0 and processed_results:
                print_sample_results(processed_results[-3:], f"{dataset_name} (Text)")
    
    # Process multimodal questions in batches
    if multimodal_questions:
        print(f"Processing {len(multimodal_questions)} multimodal questions in batches of {BATCH_SIZE}")
        mm_idx = 0
        
        for i in tqdm(range(0, len(multimodal_questions), BATCH_SIZE), desc=f"Multimodal batches for {dataset_name}"):
            batch = multimodal_questions[i:i+BATCH_SIZE]
            
            # Skip empty batches
            if not batch:
                continue
                
            try:
                batch_responses = process_multimodal_batch(batch)
                
                # Map responses back to original indices
                for response in batch_responses:
                    # Find the corresponding original index
                    while mm_idx < len(question_indices) and not question_indices[mm_idx][1]:  # Skip text-only
                        mm_idx += 1
                    if mm_idx < len(question_indices):
                        all_responses[question_indices[mm_idx][0]] = response
                        
                        # Create result entry for sample printing
                        orig_idx = question_indices[mm_idx][0]
                        question_obj = questions[orig_idx]
                        parsed_response = extract_json_from_response(response)
                        
                        result_entry = {
                            "id": question_obj.get('id', orig_idx),
                            "question": question_obj.get('question', ''),
                            "response": parsed_response,
                            "has_images": True
                        }
                        processed_results.append(result_entry)
                        mm_idx += 1
                        
            except Exception as e:
                print(f"Error processing multimodal batch {i//BATCH_SIZE + 1}: {str(e)}")
                # Fill with error responses
                for j in range(len(batch)):
                    while mm_idx < len(question_indices) and not question_indices[mm_idx][1]:
                        mm_idx += 1
                    if mm_idx < len(question_indices):
                        all_responses[question_indices[mm_idx][0]] = f"Batch processing error: {str(e)}"
                        mm_idx += 1
            
            # Print sample results every 5 batches for multimodal (they're typically slower)
            if (i//BATCH_SIZE + 1) % 5 == 0 and processed_results:
                print_sample_results([r for r in processed_results if r['has_images']][-2:], f"{dataset_name} (Multimodal)")
    
    return all_responses

# Process each dataset
for dataset_name, filename in dataset_files.items():
    print(f"\n{'='*50}")
    print(f"Processing {dataset_name} dataset")
    print(f"{'='*50}")
    
    # Load dataset
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Warning: File {filename} not found. Skipping {dataset_name}")
        continue
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in {filename}. Skipping {dataset_name}")
        continue
    
    # Initialize results for this dataset
    results[dataset_name] = []
    
    # Handle different dataset structures
    if isinstance(dataset, dict):
        # If dataset is a dictionary, look for the main data
        if dataset_name.lower() in dataset:
            questions = dataset[dataset_name.lower()]
        elif 'data' in dataset:
            questions = dataset['data']
        else:
            # Take the first key that contains a list
            for key, value in dataset.items():
                if isinstance(value, list):
                    questions = value
                    break
            else:
                print(f"Warning: Could not find questions in {dataset_name}. Skipping.")
                continue
    else:
        questions = dataset
    
    # Count image-based vs text-based questions
    total_questions = len(questions)
    image_based_count = sum(1 for q in questions if has_images(q))
    text_based_count = total_questions - image_based_count
    
    print(f"Total questions: {total_questions}")
    print(f"Image-based questions: {image_based_count}")
    print(f"Text-based questions: {text_based_count}")
    
    # Process questions in batches
    try:
        all_responses = process_questions_in_batches(questions, dataset_name)
        
        # Create result entries
        for idx, (question_obj, response) in enumerate(zip(questions, all_responses)):
            question_text = question_obj.get('question', '')
            question_id = question_obj.get('id', idx)
            
            if response is not None:
                # Extract JSON from response
                parsed_response = extract_json_from_response(response)
            else:
                parsed_response = {"error": "No response generated"}
            
            # Create result entry
            result_entry = {
                "id": question_id,
                "question": question_text,
                "response": parsed_response,
                "has_images": has_images(question_obj)
            }
            
            results[dataset_name].append(result_entry)
            
    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {str(e)}")
        # Add error entries for all questions
        for idx, question_obj in enumerate(questions):
            error_entry = {
                "id": question_obj.get('id', idx),
                "question": question_obj.get('question', ''),
                "response": {"error": f"Dataset processing failed: {str(e)}"},
                "has_images": has_images(question_obj)
            }
            results[dataset_name].append(error_entry)
    
    print(f"Completed {dataset_name}: {len(results[dataset_name])} questions processed")

# Save results
output_dir = "results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save individual dataset results
for dataset_name in results:
    if results[dataset_name]:  # Only save if there are results
        filename = os.path.join(output_dir, f"{dataset_name.lower()}_results.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({dataset_name: results[dataset_name]}, f, indent=2, ensure_ascii=False)
        print(f"Results for {dataset_name} saved to {filename}")

# Save combined results
combined_filename = os.path.join(output_dir, "all_datasets_results.json")
with open(combined_filename, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nAll results saved to {combined_filename}")

# Print summary statistics
print(f"\n{'='*50}")
print("SUMMARY STATISTICS")
print(f"{'='*50}")

total_questions_processed = 0
total_image_questions = 0
total_text_questions = 0

for dataset_name in results:
    if results[dataset_name]:
        dataset_total = len(results[dataset_name])
        dataset_images = sum(1 for q in results[dataset_name] if q.get('has_images', False))
        dataset_text = dataset_total - dataset_images
        
        print(f"{dataset_name}:")
        print(f"  Total: {dataset_total}")
        print(f"  Image-based: {dataset_images}")
        print(f"  Text-based: {dataset_text}")
        
        total_questions_processed += dataset_total
        total_image_questions += dataset_images
        total_text_questions += dataset_text

print(f"\nOVERALL TOTALS:")
print(f"Total questions processed: {total_questions_processed}")
print(f"Total image-based questions: {total_image_questions}")
print(f"Total text-based questions: {total_text_questions}")
print(f"Percentage image-based: {(total_image_questions/total_questions_processed*100):.2f}%")
print(f"Percentage text-based: {(total_text_questions/total_questions_processed*100):.2f}%")

print("\nBatch inference completed successfully!")