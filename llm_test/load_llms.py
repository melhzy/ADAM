"""
Complete Unsloth Model Comparison System for Jupyter Notebook
Compares output quality between different Unsloth 4-bit models
Designed for RTX 4090 Mobile GPU (16GB VRAM)
"""

# ============================================
# IMPORTS
# ============================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import gc
import warnings
import numpy as np
from typing import Dict, List, Optional, Tuple
import re
from dataclasses import dataclass
from IPython.display import display, HTML, clear_output
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================
# DATA CLASSES
# ============================================

@dataclass
class ModelConfig:
    """Configuration for a model"""
    model_id: str
    huggingface_path: str
    display_name: str
    size_gb: float
    prompt_template: str

@dataclass
class QueryResult:
    """Result from a model query"""
    model_id: str
    question: str
    answer: str
    generation_time: float
    tokens_generated: int
    tokens_per_second: float
    
@dataclass
class QualityMetrics:
    """Quality metrics for text evaluation"""
    word_count: int
    sentence_count: int
    vocabulary_diversity: float
    relevance_score: float
    completeness_score: float
    coherence_score: float
    overall_score: float

# ============================================
# MODEL CONFIGURATIONS
# ============================================

MODEL_CONFIGS = {
    "llama-8b": ModelConfig(
        model_id="llama-8b",
        huggingface_path="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        display_name="Llama 3.1 8B (4-bit)",
        size_gb=5.0,
        prompt_template="llama3"
    ),
    "llama-3b": ModelConfig(
        model_id="llama-3b",
        huggingface_path="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        display_name="Llama 3.2 3B (4-bit)",
        size_gb=2.0,
        prompt_template="llama3"
    ),
    "tinyllama": ModelConfig(
        model_id="tinyllama",
        huggingface_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        display_name="TinyLlama 1.1B",
        size_gb=1.0,
        prompt_template="tinyllama"
    ),
    "mistral": ModelConfig(
        model_id="mistral",
        huggingface_path="unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
        display_name="Mistral 7B (4-bit)",
        size_gb=4.0,
        prompt_template="mistral"
    )
}

# ============================================
# PROMPT TEMPLATES
# ============================================

class PromptTemplates:
    """Prompt templates for different model types"""
    
    @staticmethod
    def get_prompt(template_type: str, question: str) -> str:
        """Get formatted prompt for a specific model type"""
        
        if template_type == "llama3":
            return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that provides accurate and detailed information.<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        elif template_type == "tinyllama":
            return f"""<|system|>
You are a helpful assistant.</s>
<|user|>
{question}</s>
<|assistant|>
"""
        
        elif template_type == "mistral":
            return f"[INST] {question} [/INST]"
        
        else:  # Generic
            return f"Question: {question}\n\nAnswer: "

# ============================================
# MODEL LOADER CLASS
# ============================================

class UnslothModelSystem:
    """Main system for loading and querying Unsloth models"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_model = None
        self.current_tokenizer = None
        self.current_config = None
        self._show_system_info()
    
    def _show_system_info(self):
        """Display system information"""
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name()
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🖥️ System: {gpu_name} with {vram_gb:.2f}GB VRAM")
        else:
            print("⚠️ No GPU detected, using CPU (will be slow)")
    
    def load_model(self, model_id: str) -> bool:
        """Load a specific model"""
        
        # Clean up any existing model
        self._cleanup()
        
        if model_id not in MODEL_CONFIGS:
            print(f"❌ Unknown model ID: {model_id}")
            print(f"Available: {list(MODEL_CONFIGS.keys())}")
            return False
        
        config = MODEL_CONFIGS[model_id]
        print(f"\n📦 Loading {config.display_name}...")
        print(f"Path: {config.huggingface_path}")
        
        try:
            # Load tokenizer
            print("Loading tokenizer...")
            self.current_tokenizer = AutoTokenizer.from_pretrained(
                config.huggingface_path,
                trust_remote_code=True
            )
            
            # Set padding token
            if self.current_tokenizer.pad_token is None:
                self.current_tokenizer.pad_token = self.current_tokenizer.eos_token
            
            # Load model
            print("Loading model weights...")
            self.current_model = AutoModelForCausalLM.from_pretrained(
                config.huggingface_path,
                device_map="auto",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            self.current_config = config
            
            # Check memory usage
            if self.device == "cuda":
                allocated_gb = torch.cuda.memory_allocated() / 1e9
                print(f"✅ Model loaded! Using {allocated_gb:.2f}GB VRAM")
            else:
                print(f"✅ Model loaded on CPU")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {str(e)[:200]}")
            self._cleanup()
            return False
    
    def query(self, question: str, max_tokens: int = 256, temperature: float = 0.7) -> Optional[QueryResult]:
        """Query the current model"""
        
        if self.current_model is None:
            print("❌ No model loaded")
            return None
        
        # Get prompt
        prompt = PromptTemplates.get_prompt(self.current_config.prompt_template, question)
        
        # Tokenize
        inputs = self.current_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        print(f"Generating response...")
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.current_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.15,
                pad_token_id=self.current_tokenizer.pad_token_id,
                eos_token_id=self.current_tokenizer.eos_token_id,
            )
        
        generation_time = time.time() - start_time
        
        # Decode response
        full_response = self.current_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (remove prompt)
        if prompt in full_response:
            answer = full_response[len(prompt):].strip()
        else:
            # Try to extract after common markers
            answer = full_response
            for marker in ["Answer:", "Assistant:", "assistant", "</s>", "<|assistant|>"]:
                if marker in answer:
                    parts = answer.split(marker)
                    if len(parts) > 1:
                        answer = parts[-1].strip()
                        break
        
        # Calculate metrics
        input_tokens = inputs['input_ids'].shape[1]
        output_tokens = outputs.shape[1] - input_tokens
        tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0
        
        return QueryResult(
            model_id=self.current_config.model_id,
            question=question,
            answer=answer,
            generation_time=generation_time,
            tokens_generated=output_tokens,
            tokens_per_second=tokens_per_second
        )
    
    def _cleanup(self):
        """Clean up model and free memory"""
        if self.current_model:
            del self.current_model
        if self.current_tokenizer:
            del self.current_tokenizer
        self.current_model = None
        self.current_tokenizer = None
        self.current_config = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

# ============================================
# QUALITY EVALUATOR
# ============================================

class QualityEvaluator:
    """Evaluate text quality without external dependencies"""
    
    @staticmethod
    def evaluate(text: str, question: str) -> QualityMetrics:
        """Evaluate text quality"""
        
        # Basic counts
        words = text.split()
        word_count = len(words)
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Vocabulary diversity
        unique_words = set(w.lower() for w in words)
        vocab_diversity = len(unique_words) / word_count if word_count > 0 else 0
        
        # Relevance score
        relevance = QualityEvaluator._calculate_relevance(text, question)
        
        # Completeness score
        completeness = QualityEvaluator._calculate_completeness(text, question)
        
        # Coherence score
        coherence = QualityEvaluator._calculate_coherence(sentences)
        
        # Overall score
        overall = (relevance * 0.3 + completeness * 0.3 + 
                  vocab_diversity * 0.2 + coherence * 0.2)
        
        return QualityMetrics(
            word_count=word_count,
            sentence_count=sentence_count,
            vocabulary_diversity=vocab_diversity,
            relevance_score=relevance,
            completeness_score=completeness,
            coherence_score=coherence,
            overall_score=overall
        )
    
    @staticmethod
    def _calculate_relevance(text: str, question: str) -> float:
        """Calculate how relevant the answer is to the question"""
        text_lower = text.lower()
        question_lower = question.lower()
        
        # Extract key terms from question
        question_words = set(re.findall(r'\b\w+\b', question_lower))
        stop_words = {'what', 'is', 'where', 'the', 'a', 'an', 'and', 'or'}
        question_words = question_words - stop_words
        
        if not question_words:
            return 0.5
        
        # Count how many question words appear in answer
        matches = sum(1 for word in question_words if word in text_lower)
        return matches / len(question_words)
    
    @staticmethod
    def _calculate_completeness(text: str, question: str) -> float:
        """Calculate if the answer is complete"""
        score = 0.0
        text_lower = text.lower()
        
        # For Mars question specifically
        if 'mars' in question.lower():
            # Check for "what is Mars" part
            if 'mars' in text_lower:
                score += 0.25
            if any(w in text_lower for w in ['planet', 'celestial', 'body']):
                score += 0.25
            
            # Check for "where is Mars" part  
            if any(w in text_lower for w in ['fourth', '4th', 'solar system']):
                score += 0.25
            if any(w in text_lower for w in ['sun', 'orbit', 'earth', 'jupiter']):
                score += 0.25
        
        return min(1.0, score)
    
    @staticmethod
    def _calculate_coherence(sentences: List[str]) -> float:
        """Calculate text coherence"""
        if len(sentences) < 2:
            return 1.0
        
        # Simple coherence: check word overlap between sentences
        coherence_scores = []
        for i in range(len(sentences) - 1):
            if sentences[i].strip() and sentences[i+1].strip():
                words1 = set(sentences[i].lower().split())
                words2 = set(sentences[i+1].lower().split())
                if words1 and words2:
                    overlap = len(words1 & words2) / min(len(words1), len(words2))
                    coherence_scores.append(min(1.0, overlap * 2))
        
        return np.mean(coherence_scores) if coherence_scores else 0.5

# ============================================
# COMPARISON SYSTEM
# ============================================

class ModelComparison:
    """Compare multiple models"""
    
    def __init__(self):
        self.system = UnslothModelSystem()
        self.evaluator = QualityEvaluator()
        self.results = {}
    
    def compare_models(self, model_ids: List[str], question: str, max_tokens: int = 300) -> pd.DataFrame:
        """Compare multiple models on the same question"""
        
        results_data = []
        
        for model_id in model_ids:
            print(f"\n{'='*70}")
            print(f"Testing: {MODEL_CONFIGS[model_id].display_name}")
            print('='*70)
            
            # Load model
            if not self.system.load_model(model_id):
                print(f"Skipping {model_id} due to loading error")
                continue
            
            # Query model
            result = self.system.query(question, max_tokens=max_tokens)
            if result is None:
                print(f"No result from {model_id}")
                continue
            
            # Evaluate quality
            metrics = self.evaluator.evaluate(result.answer, question)
            
            # Store results
            results_data.append({
                'Model': MODEL_CONFIGS[model_id].display_name,
                'Words': metrics.word_count,
                'Sentences': metrics.sentence_count,
                'Vocab Diversity': f"{metrics.vocabulary_diversity:.3f}",
                'Relevance': f"{metrics.relevance_score:.3f}",
                'Completeness': f"{metrics.completeness_score:.3f}",
                'Coherence': f"{metrics.coherence_score:.3f}",
                'Overall Score': f"{metrics.overall_score:.3f}",
                'Gen Time (s)': f"{result.generation_time:.2f}",
                'Tokens/sec': f"{result.tokens_per_second:.1f}",
                'Answer': result.answer[:200] + "..." if len(result.answer) > 200 else result.answer
            })
            
            # Clean up before next model
            time.sleep(1)
        
        # Create comparison dataframe
        df = pd.DataFrame(results_data)
        
        return df
    
    def display_comparison(self, df: pd.DataFrame):
        """Display comparison results in HTML"""
        
        # Create HTML table
        html = """
        <h2>📊 Model Comparison Results</h2>
        <style>
            .comparison-table {
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }
            .comparison-table th {
                background-color: #4CAF50;
                color: white;
                padding: 12px;
                text-align: left;
            }
            .comparison-table td {
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }
            .comparison-table tr:hover {
                background-color: #f5f5f5;
            }
            .answer-cell {
                max-width: 400px;
                font-size: 0.9em;
            }
        </style>
        <table class="comparison-table">
        """
        
        # Add header
        html += "<tr>"
        for col in df.columns:
            if col != 'Answer':
                html += f"<th>{col}</th>"
        html += "</tr>"
        
        # Add rows
        for _, row in df.iterrows():
            html += "<tr>"
            for col in df.columns:
                if col != 'Answer':
                    html += f"<td>{row[col]}</td>"
            html += "</tr>"
        
        html += "</table>"
        
        # Display answers separately
        html += "<h3>📝 Generated Answers</h3>"
        for _, row in df.iterrows():
            html += f"""
            <div style="margin: 15px 0; padding: 15px; background: #f8f9fa; border-left: 4px solid #4CAF50;">
                <h4>{row['Model']}</h4>
                <p>{row['Answer']}</p>
            </div>
            """
        
        display(HTML(html))

# ============================================
# MAIN USAGE FUNCTIONS
# ============================================

def run_comparison(models_to_compare=None, question="What is Mars and where is Mars?"):
    """Main function to run model comparison"""
    
    if models_to_compare is None:
        # Default models to compare
        models_to_compare = ["llama-8b", "llama-3b"]
    
    print(f"🚀 Starting Model Comparison")
    print(f"Question: {question}")
    print(f"Models: {models_to_compare}")
    
    # Create comparison system
    comparison = ModelComparison()
    
    # Run comparison
    df = comparison.compare_models(models_to_compare, question)
    
    # Display results
    comparison.display_comparison(df)
    
    # Return dataframe for further analysis
    return df

def quick_test(model_id="llama-3b", question="What is Mars?"):
    """Quick test of a single model"""
    
    system = UnslothModelSystem()
    
    if system.load_model(model_id):
        result = system.query(question, max_tokens=200)
        if result:
            print(f"\n📝 Answer: {result.answer}")
            print(f"⏱️ Time: {result.generation_time:.2f}s")
            print(f"⚡ Speed: {result.tokens_per_second:.1f} tokens/sec")
            return result
    
    return None

# ============================================
# JUPYTER NOTEBOOK USAGE
# ============================================

if __name__ == "__main__":
    print("Unsloth Model Comparison System")
    print("=" * 70)
    print("\nUsage Examples:")
    print("1. Compare models: df = run_comparison(['llama-8b', 'llama-3b'])")
    print("2. Quick test: result = quick_test('llama-3b')")
    print("3. Custom comparison: run_comparison(['tinyllama', 'mistral'], 'Your question here')")
    print("\nAvailable models:", list(MODEL_CONFIGS.keys()))