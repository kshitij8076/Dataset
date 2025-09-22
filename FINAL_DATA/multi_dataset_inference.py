import time
import torch
import json
from transformers import AutoTokenizer, AutoProcessor, Llama4ForConditionalGeneration
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
model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

print("Loading tokenizer and processor...")
tokenizer = AutoTokenizer.from_pretrained(model_id)     # used for text-only inference
processor = AutoProcessor.from_pretrained(model_id)     # used for multimodal inference

print("Loading model...")
model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    attn_implementation="sdpa",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

print("Model loaded successfully!")

# Initialize results dictionary
results = {}

# JSON formatting instruction
instruction = """
Please analyze this question and image (if given); provide your response in the following JSON format:
{
  "answer": "Your answer here (numerical value or option letter for MCQ, or null for proof/open-ended questions)",
  "solution": "Detailed step-by-step solution"
}
Return ONLY valid JSON without any additional text.
"""

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

def process_text_question(question_text):
    """Process text-only question"""
    messages = [
        {
            "role": "user",
            "content": f"{question_text}\n\n{instruction}"
        }
    ]
    
    # Use tokenizer for text-only questions
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(model.device)
    
    outputs = model.generate(
        inputs,
        max_new_tokens=8196,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.batch_decode(outputs[:, inputs.shape[-1]:])[0]
    return response

def process_multimodal_question(question_text, images):
    """Process question with images"""
    # Construct content with images
    content = []
    
    # Add images first
    for img_path in images:
        content.append({"type": "image", "url": img_path})
    
    # Add the question text
    content.append({"type": "text", "text": f"{question_text}\n\n{instruction}"})
    
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    # Use processor for multimodal questions
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=8196,
    )
    
    response = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
    return response

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
    
    # Process each question
    for idx, question_obj in enumerate(tqdm(questions, desc=f"Processing {dataset_name}")):
        try:
            question_text = question_obj.get('question', '')
            question_id = question_obj.get('id', idx)
            
            # Check if question has images
            if has_images(question_obj):
                # Process multimodal question
                images = question_obj['image']
                if isinstance(images, str):
                    images = [images]
                
                response = process_multimodal_question(question_text, images)
            else:
                # Process text-only question
                response = process_text_question(question_text)
            
            # Extract JSON from response
            parsed_response = extract_json_from_response(response)
            
            # Create result entry
            result_entry = {
                "id": question_id,
                "question": question_text,
                "response": parsed_response,
                "has_images": has_images(question_obj)
            }
            
            results[dataset_name].append(result_entry)
            
        except Exception as e:
            print(f"Error processing question {idx} in {dataset_name}: {str(e)}")
            # Add error entry
            error_entry = {
                "id": question_obj.get('id', idx),
                "question": question_obj.get('question', ''),
                "response": {"error": f"Processing failed: {str(e)}"},
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

print("\nInference completed successfully!")