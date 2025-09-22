import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig
)
import json
import pandas as pd
from tqdm import tqdm
import logging
from typing import List, Dict, Any, Optional
import argparse
from datetime import datetime
import os
import re
import gc
import psutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GPUStats:
    """Track GPU memory usage"""
    device_id: int
    memory_allocated: float
    memory_reserved: float
    memory_free: float

class OptimizedQuestionInferrerProcessor:
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.3-70B-Instruct", 
                 batch_size: int = 128,  # Increased for A100s
                 use_quantization: bool = False,  # Disable for A100s with enough memory
                 max_workers: int = 8):  # For parallel processing
        """
        Initialize the Optimized Question Inferrer processor for multi-GPU setup
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # Check GPU availability and setup
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. This optimized version requires GPUs.")
        
        self.num_gpus = torch.cuda.device_count()
        logger.info(f"Detected {self.num_gpus} GPUs")
        
        # Print GPU information
        for i in range(self.num_gpus):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            logger.info(f"GPU {i}: {props.name} - {memory_gb:.1f}GB")
        
        # Set memory allocation strategy
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        
        # Define valid subfields
        self.physics_subfields = [
            "Mechanics", "Thermodynamics", "Electromagnetism", "Optics", 
            "Modern Physics", "Waves", "Fluid Mechanics", "Nuclear Physics"
        ]
        
        self.maths_subfields = [
            "Algebra", "Geometry", "Calculus", "Statistics", "Trigonometry", 
            "Number Theory", "Discrete Mathematics", "Linear Algebra", "Probability"
        ]
        
        # Disable quantization for A100s with sufficient memory
        if use_quantization:
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,  # Better for A100
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            logger.info("Using 4-bit quantization")
        else:
            self.quantization_config = None
            logger.info("Using full precision (recommended for A100s)")
            
        self.tokenizer = None
        self.model = None
        self.pipe = None
        
    def get_gpu_stats(self) -> List[GPUStats]:
        """Get current GPU memory statistics"""
        stats = []
        for i in range(self.num_gpus):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory / (1024**3)
            free = total - reserved
            
            stats.append(GPUStats(i, allocated, reserved, free))
        return stats
    
    def print_gpu_stats(self):
        """Print current GPU memory usage"""
        stats = self.get_gpu_stats()
        logger.info("GPU Memory Usage:")
        for stat in stats:
            logger.info(f"  GPU {stat.device_id}: {stat.memory_allocated:.1f}GB allocated, "
                       f"{stat.memory_reserved:.1f}GB reserved, {stat.memory_free:.1f}GB free")

    def aggressive_memory_cleanup(self):
        """Aggressively clean up GPU memory"""
        # Clear PyTorch cache
        for i in range(self.num_gpus):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
        
        # Additional cleanup
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            for i in range(self.num_gpus):
                torch.cuda.reset_peak_memory_stats(i)

    def load_model(self):
        """Load the tokenizer and model with optimized multi-GPU setup"""
        logger.info(f"Loading tokenizer for {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",  # Better for batch generation
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info(f"Loading model {self.model_name} across {self.num_gpus} GPUs")
        
        # Optimized model loading kwargs for A100s
        model_kwargs = {
            "torch_dtype": torch.bfloat16,  # Better for A100
            "device_map": "auto",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        if self.quantization_config:
            model_kwargs["quantization_config"] = self.quantization_config
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Create optimized pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            return_full_text=False,
            pad_token_id=self.tokenizer.eos_token_id,
            batch_size=self.batch_size,
            # Pipeline optimizations
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            max_new_tokens=150,
        )
        
        logger.info("Model loaded successfully")
        self.print_gpu_stats()
        
    def create_system_prompt(self) -> str:
        """Create the system prompt for question inference"""
        physics_fields = ", ".join(self.physics_subfields)
        maths_fields = ", ".join(self.maths_subfields)
        
        return f"""You are a Question Inferrer AI. Your task is to analyze educational questions and return ONLY a JSON response with specific categorization.

INSTRUCTIONS:
1. Analyze the given question and answer
2. Return ONLY a valid JSON object with these exact fields
3. Do NOT include any other text, explanations, or formatting

CATEGORIZATION RULES:

Subject:
- "PHYSICS" if related to physics concepts
- "MATHS" if related to mathematics concepts  
- "EXT" if belongs to any other subject

Type:
- "MCQ" for multiple choice questions
- "Numerical" for questions requiring numerical calculations
- "Open Ended" for descriptive/theoretical questions
- "EXT" for any other question type

Subfield (Physics): {physics_fields}
Subfield (Maths): {maths_fields}
- Use "EXT" if subject is not Physics/Maths or subfield doesn't match

Difficulty:
- "Level 1" (Foundational - Elementary School, Grades 3-5): Basic factual recall, simple arithmetic
- "Level 2" (Intermediate - Middle School, Grades 6-8): Multi-step reasoning, basic problem-solving
- "Level 3" (Proficient - High School, Grades 9-12): Conceptual understanding, abstract thinking
- "Level 4" (Advanced - Undergraduate): In-depth problem-solving, theory-based analysis
- "Level 5" (Expert - Graduate/Research): Novel reasoning, cross-domain synthesis

REQUIRED JSON FORMAT:
{{
    "subject": "PHYSICS|MATHS|EXT",
    "type": "MCQ|Numerical|Open Ended|EXT", 
    "subfield": "specific_subfield_or_EXT",
    "difficulty": "Level 1|Level 2|Level 3|Level 4|Level 5"
}}

Return ONLY the JSON object, nothing else."""

    def format_question_prompt(self, question: str, answer: str) -> str:
        """Format the question and answer into a prompt"""
        system_prompt = self.create_system_prompt()
        
        user_prompt = f"""Question: {question}

Answer: {answer}

Analyze and return JSON:"""

        full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return full_prompt

    def extract_json_from_response(self, response: str) -> Optional[Dict]:
        """Extract and validate JSON from model response"""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                json_str = json_match.group()
                parsed_json = json.loads(json_str)
                
                # Validate required fields
                required_fields = ["subject", "type", "subfield", "difficulty"]
                if all(field in parsed_json for field in required_fields):
                    return parsed_json
                    
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse JSON from response: {e}")
            
        return None

    def process_batch_optimized(self, questions: List[str], answers: List[str]) -> List[Dict]:
        """Optimized batch processing with aggressive memory management"""
        start_time = time.time()
        
        # Pre-format all prompts
        prompts = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            prompt_futures = [
                executor.submit(self.format_question_prompt, q, a) 
                for q, a in zip(questions, answers)
            ]
            prompts = [future.result() for future in prompt_futures]
        
        logger.info(f"Formatted {len(prompts)} prompts in {time.time() - start_time:.2f}s")
        
        try:
            # Generate responses with optimized settings
            generation_start = time.time()
            responses = self.pipe(
                prompts,
                max_new_tokens=150,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                batch_size=self.batch_size,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False,
                clean_up_tokenization_spaces=True,
            )
            
            generation_time = time.time() - generation_start
            logger.info(f"Generated {len(responses)} responses in {generation_time:.2f}s "
                       f"({len(responses)/generation_time:.1f} samples/s)")
            
            # Process responses in parallel
            processing_start = time.time()
            results = []
            
            def process_single_response(i, response):
                if isinstance(response, list):
                    generated_text = response[0]['generated_text'].strip()
                else:
                    generated_text = response['generated_text'].strip()
                
                parsed_json = self.extract_json_from_response(generated_text)
                
                if parsed_json:
                    return {
                        "question": questions[i],
                        "answer": answers[i],
                        "inference": parsed_json,
                        "raw_response": generated_text,
                        "status": "success"
                    }
                else:
                    return {
                        "question": questions[i],
                        "answer": answers[i],
                        "inference": {
                            "subject": "EXT",
                            "type": "EXT", 
                            "subfield": "EXT",
                            "difficulty": "Level 1"
                        },
                        "raw_response": generated_text,
                        "status": "failed_parsing"
                    }
            
            # Parallel response processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                result_futures = [
                    executor.submit(process_single_response, i, response)
                    for i, response in enumerate(responses)
                ]
                results = [future.result() for future in result_futures]
            
            processing_time = time.time() - processing_start
            logger.info(f"Processed responses in {processing_time:.2f}s")
            
            # Aggressive memory cleanup
            del prompts, responses
            self.aggressive_memory_cleanup()
            
            total_time = time.time() - start_time
            logger.info(f"Total batch time: {total_time:.2f}s ({len(results)/total_time:.1f} samples/s)")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            # Clean up memory even on error
            self.aggressive_memory_cleanup()
            
            # Return fallback results
            return [{
                "question": q,
                "answer": a,
                "inference": {
                    "subject": "EXT",
                    "type": "EXT",
                    "subfield": "EXT", 
                    "difficulty": "Level 1"
                },
                "raw_response": "",
                "status": "error"
            } for q, a in zip(questions, answers)]

    def load_json_dataset(self, file_path: str) -> List[Dict]:
        """Load dataset from JSON file with parallel processing"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                data = [data]
            
            logger.info(f"Loaded {len(data)} samples from {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []

    def process_dataset_file_optimized(self, file_path: str, question_key: str = "question", 
                                     answer_key: str = "answer") -> List[Dict]:
        """Optimized dataset file processing"""
        dataset = self.load_json_dataset(file_path)
        if not dataset:
            return []
        
        results = []
        total_batches = (len(dataset) + self.batch_size - 1) // self.batch_size
        
        logger.info(f"Processing {len(dataset)} samples in {total_batches} batches of {self.batch_size}")
        
        # Process in batches with progress tracking
        for batch_idx in tqdm(range(0, len(dataset), self.batch_size), 
                             desc=f"Processing {os.path.basename(file_path)}",
                             total=total_batches):
            
            batch_start_time = time.time()
            batch_end = min(batch_idx + self.batch_size, len(dataset))
            batch_data = dataset[batch_idx:batch_end]
            
            # Extract questions and answers from batch in parallel
            def extract_qa(item):
                # Extract question
                question = str(item.get(question_key, "Unknown question"))
                
                # Extract answer - try multiple possible keys
                answer = ""
                for key in [answer_key, "solution", "answer"]:
                    if key in item:
                        answer = str(item[key])
                        break
                if not answer:
                    answer = "Unknown answer"
                
                return question, answer, item
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                extraction_results = list(executor.map(extract_qa, batch_data))
            
            questions = [r[0] for r in extraction_results]
            answers = [r[1] for r in extraction_results]
            batch_metadata = [r[2] for r in extraction_results]
            
            # Process batch
            batch_results = self.process_batch_optimized(questions, answers)
            
            # Add original metadata to results
            for j, result in enumerate(batch_results):
                original_item = batch_metadata[j]
                result['original_data'] = original_item
                if 'id' in original_item:
                    result['id'] = original_item['id']
            
            results.extend(batch_results)
            
            batch_time = time.time() - batch_start_time
            samples_per_sec = len(batch_results) / batch_time
            
            logger.info(f"Batch {batch_idx//self.batch_size + 1}/{total_batches}: "
                       f"{len(batch_results)} samples in {batch_time:.2f}s "
                       f"({samples_per_sec:.1f} samples/s)")
            
            # Print GPU stats every 10 batches
            if (batch_idx // self.batch_size + 1) % 10 == 0:
                self.print_gpu_stats()
        
        return results

    def process_multiple_datasets_optimized(self, dataset_paths: List[str], 
                                          question_key: str = "question", 
                                          answer_key: str = "answer",
                                          save_results: bool = True) -> Dict[str, List[Dict]]:
        """Optimized processing of multiple dataset files"""
        all_results = {}
        
        for file_path in dataset_paths:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                continue
                
            logger.info(f"Processing dataset: {file_path}")
            start_time = time.time()
            
            results = self.process_dataset_file_optimized(file_path, question_key, answer_key)
            
            processing_time = time.time() - start_time
            if results:
                samples_per_sec = len(results) / processing_time
                logger.info(f"Completed {file_path}: {len(results)} samples in "
                           f"{processing_time:.2f}s ({samples_per_sec:.1f} samples/s)")
            
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            all_results[file_name] = results
            
            if save_results:
                self.save_results(results, file_name)
            
            # Memory cleanup between datasets
            self.aggressive_memory_cleanup()
        
        return all_results

    def save_results(self, results: List[Dict], dataset_name: str):
        """Save results to JSON and CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        json_filename = f"question_inference_{dataset_name}_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed results saved to {json_filename}")
        
        # Create summary CSV with ID preservation
        summary_data = []
        for result in results:
            inference = result['inference']
            summary_row = {
                'id': result.get('id', 'N/A'),
                'question': result['question'][:100] + '...' if len(result['question']) > 100 else result['question'],
                'subject': inference['subject'],
                'type': inference['type'],
                'subfield': inference['subfield'],
                'difficulty': inference['difficulty'],
                'status': result['status']
            }
            summary_data.append(summary_row)
        
        csv_filename = f"question_inference_summary_{dataset_name}_{timestamp}.csv"
        df = pd.DataFrame(summary_data)
        df.to_csv(csv_filename, index=False)
        logger.info(f"Summary saved to {csv_filename}")
        
        # Create inference-only JSON for easy integration
        inference_only = []
        for result in results:
            inference_item = {
                'id': result.get('id', 'N/A'),
                'inference': result['inference']
            }
            inference_only.append(inference_item)
        
        inference_filename = f"inference_only_{dataset_name}_{timestamp}.json"
        with open(inference_filename, 'w', encoding='utf-8') as f:
            json.dump(inference_only, f, indent=2, ensure_ascii=False)
        logger.info(f"Inference-only results saved to {inference_filename}")
        
        # Print statistics
        self.print_statistics(results, dataset_name)

    def print_statistics(self, results: List[Dict], dataset_name: str):
        """Print processing statistics"""
        total = len(results)
        if total == 0:
            return
            
        successful = sum(1 for r in results if r['status'] == 'success')
        failed_parsing = sum(1 for r in results if r['status'] == 'failed_parsing')
        errors = sum(1 for r in results if r['status'] == 'error')
        
        subjects = {}
        types = {}
        difficulties = {}
        
        for result in results:
            inf = result['inference']
            subjects[inf['subject']] = subjects.get(inf['subject'], 0) + 1
            types[inf['type']] = types.get(inf['type'], 0) + 1
            difficulties[inf['difficulty']] = difficulties.get(inf['difficulty'], 0) + 1
        
        print(f"\n=== Statistics for {dataset_name} ===")
        print(f"Total processed: {total}")
        print(f"Successful: {successful} ({successful/total*100:.1f}%)")
        print(f"Failed parsing: {failed_parsing} ({failed_parsing/total*100:.1f}%)")
        print(f"Errors: {errors} ({errors/total*100:.1f}%)")
        
        print(f"\nSubject distribution: {subjects}")
        print(f"Type distribution: {types}")
        print(f"Difficulty distribution: {difficulties}")

def main():
    parser = argparse.ArgumentParser(description="Optimized Question Inferrer Batch Processing")
    parser.add_argument("--datasets", nargs='+', required=True, 
                       help="Paths to JSON dataset files")
    parser.add_argument("--question_key", default="question", 
                       help="Key name for question field in JSON")
    parser.add_argument("--answer_key", default="answer", 
                       help="Key name for answer field in JSON")
    parser.add_argument("--batch_size", type=int, default=128, 
                       help="Batch size for processing (increased for A100s)")
    parser.add_argument("--use_quantization", action="store_true", 
                       help="Enable 4-bit quantization (not recommended for A100s)")
    parser.add_argument("--max_workers", type=int, default=8,
                       help="Number of worker threads for parallel processing")
    
    args = parser.parse_args()
    
    # Initialize optimized processor
    processor = OptimizedQuestionInferrerProcessor(
        batch_size=args.batch_size,
        use_quantization=args.use_quantization,
        max_workers=args.max_workers
    )
    
    # Load model
    processor.load_model()
    
    # Process datasets
    all_results = processor.process_multiple_datasets_optimized(
        dataset_paths=args.datasets,
        question_key=args.question_key,
        answer_key=args.answer_key
    )
    
    logger.info("Processing completed!")

if __name__ == "__main__":
    # Process all your specific datasets with optimized settings
    dataset_files = [
        'AGI_EVAL.json',
        'emma_math_dataset.json', 
        'emma_physics_dataset.json',
        'gpqa_physics_dataset.json',
        'gsm8k_dataset.json',
        'jeebench_dataset.json',
        'mathvista_dataset.json',
        'mmlu_math_physics_test.json',
        'mmmu_arch_eng_test_dataset.json',
        'olympiad_bench_dataset.json',
        'scibench_dataset.json',
        'scienceqa_test_dataset.json',
        'scieval_physics_combined.json'
    ]
    
    # Initialize optimized processor with settings for 8x A100 40GB
    processor = OptimizedQuestionInferrerProcessor(
        batch_size=32,  # Aggressive batch size for A100s
        use_quantization=False,  # Full precision for better quality and speed on A100s
        max_workers=8  # Increased parallel workers
    )
    
    print("🚀 Starting optimized multi-GPU processing...")
    print(f"Hardware: {torch.cuda.device_count()}x GPUs detected")
    print(f"Batch size: {processor.batch_size}")
    print(f"Found {len(dataset_files)} datasets to process")
    
    # Load model with optimizations
    processor.load_model()
    
    # Filter existing files
    existing_files = []
    missing_files = []
    
    for file_path in dataset_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            print(f"✓ Found: {file_path}")
        else:
            missing_files.append(file_path)
            print(f"✗ Missing: {file_path}")
    
    if missing_files:
        print(f"\nWarning: {len(missing_files)} files not found")
    
    if existing_files:
        print(f"\n🚀 Processing {len(existing_files)} datasets with optimization...")
        overall_start = time.time()
        
        total_processed = 0
        
        # Process each dataset with optimizations
        for file_path in existing_files:
            try:
                print(f"\n{'='*50}")
                print(f"🔥 Processing: {file_path}")
                print(f"{'='*50}")
                
                file_start = time.time()
                
                # Process single dataset with optimizations
                results = processor.process_dataset_file_optimized(
                    file_path=file_path,
                    question_key="question",
                    answer_key="answer"
                )
                
                file_time = time.time() - file_start
                
                # Save results for this specific dataset
                dataset_name = os.path.splitext(os.path.basename(file_path))[0]
                processor.save_results(results, dataset_name)
                
                samples_per_sec = len(results) / file_time if file_time > 0 else 0
                total_processed += len(results)
                
                print(f"✅ Completed {file_path}")
                print(f"📊 Processed {len(results)} items in {file_time:.2f}s")
                print(f"⚡ Speed: {samples_per_sec:.1f} samples/second")
                
                # Show GPU stats after each dataset
                processor.print_gpu_stats()
                
            except Exception as e:
                logger.error(f"❌ Error processing {file_path}: {e}")
                continue
        
        overall_time = time.time() - overall_start
        overall_speed = total_processed / overall_time if overall_time > 0 else 0
        
        print(f"\n{'='*50}")
        print(f"🎉 ALL DATASETS PROCESSED SUCCESSFULLY!")
        print(f"{'='*50}")
        print(f"📈 Total items processed: {total_processed}")
        print(f"⏱️  Total time: {overall_time:.2f} seconds")
        print(f"🚀 Overall speed: {overall_speed:.1f} samples/second")
        print(f"💪 Average per dataset: {total_processed/len(existing_files):.0f} samples")
        
    else:
        print("\n❌ No dataset files found. Please ensure the following files exist:")
        for file_path in dataset_files:
            print(f"  - {file_path}")