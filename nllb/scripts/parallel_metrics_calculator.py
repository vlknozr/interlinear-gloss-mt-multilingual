#!/usr/bin/env python3
import json
import os
import argparse
import nltk
import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import sacrebleu
from sacrebleu.metrics import CHRF
import copy

nltk.download('punkt_tab')

# Import COMET if available
try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    print("WARNING: COMET package not available. XCOMET scores will not be calculated.")
    print("To install, run: pip install unbabel-comet")

def main():
    parser = argparse.ArgumentParser(description='Calculate metrics for parallel reference/prediction text files')
    parser.add_argument('reference_file', help='Reference text file (one sentence per line)')
    parser.add_argument('prediction_file', help='Prediction text file (one sentence per line)')
    parser.add_argument('--outdir', type=str, default=None, help='Directory to save the output file (defaults to prediction file\'s directory)')

    args = parser.parse_args()

    reference_file = args.reference_file
    prediction_file = args.prediction_file

    print(f"Processing reference: {reference_file}")
    print(f"Processing prediction: {prediction_file}...")

    # Determine output directory. If --outdir is not provided, default to the prediction file's directory
    output_dir = args.outdir if args.outdir else os.path.dirname(os.path.abspath(prediction_file))

    # Ensure the output directory exists
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Generate output filename in chosen directory
    basename = os.path.basename(prediction_file)
    output_file = os.path.join(output_dir, f"all_metrics_scored_{basename}")

    # Read the input files
    try:
        with open(reference_file, 'r', encoding='utf-8') as f:
            references = [line.strip() for line in f if line.strip()]
        
        with open(prediction_file, 'r', encoding='utf-8') as f:
            predictions = [line.strip() for line in f if line.strip()]
        
        # Check if files have same number of lines
        if len(references) != len(predictions):
            print(f"WARNING: Files have different number of lines: {len(references)} references vs {len(predictions)} predictions")
            # Use the shorter length
            min_len = min(len(references), len(predictions))
            references = references[:min_len]
            predictions = predictions[:min_len]
            print(f"Using only the first {min_len} lines from each file")
    
    except Exception as e:
        print(f"Error reading files: {e}")
        raise SystemExit(1)
    
    # Create the data structure
    data = {
        "results": []
    }
    
    # Add each reference-prediction pair to the data
    for i, (reference, prediction) in enumerate(zip(references, predictions)):
        data["results"].append({
            "id": i,
            "reference": reference,
            "prediction": prediction
        })
    
    # Calculate BLEU scores
    print("Calculating BLEU scores...")
    calculate_bleu(data)
    
    # Calculate chrF++ scores
    print("\nCalculating chrF++ scores...")
    calculate_chrf(data)
    
    # Calculate XCOMET scores if available
    if COMET_AVAILABLE:
        print("\nCalculating XCOMET scores...")
        try:
            calculate_xcomet(data)
            print("\nXCOMET scores have been calculated and added to each item.")
        except Exception as e:
            print(f"\nError calculating XCOMET scores: {e}")
            print("Failed to calculate XCOMET scores.")
    
    # Calculate summary statistics and reorganize data
    summary = calculate_summary(data)
    
    # Create the final output structure
    output_data = {
        'summary': summary,
        'results': data['results']
    }
    
    # Write the results to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, indent=2, ensure_ascii=False, fp=f)
    
    # Print summary information
    print_summary(summary)
    print(f"\nResults saved to {output_file}")

def calculate_bleu(data):
    """Calculate BLEU scores for all translations"""
    # Download necessary NLTK resources if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    # Initialize smoothing function for BLEU score
    smoothing = SmoothingFunction().method1
    
    # Prepare lists for corpus BLEU computation
    references_tokenized_corpus = []
    predictions_tokenized_corpus = []
    
    # Lists to store individual BLEU scores
    bleu_scores = []
    
    # Calculate individual BLEU scores and prepare data for corpus BLEU
    for item in data['results']:
        reference = item['reference']
        prediction = item['prediction']
        
        # Tokenize sentences
        reference_tokens = word_tokenize(reference.lower())
        
        # Add to corpus BLEU computation lists
        references_tokenized_corpus.append([reference_tokens])
        
        # Calculate and store individual BLEU scores
        prediction_tokens = word_tokenize(prediction.lower())
        predictions_tokenized_corpus.append(prediction_tokens)
        
        # Use smoothing to handle cases where there are no overlapping n-grams
        bleu = sentence_bleu([reference_tokens], prediction_tokens, smoothing_function=smoothing)
        bleu_scores.append(bleu)
        
        # Store on 0-100 scale
        item['bleu'] = round(bleu * 100, 2)
    
    # Calculate corpus BLEU score (overall BLEU)
    weights = (0.25, 0.25, 0.25, 0.25)  # Default weights for 1-4 grams
    
    corpus_bleu_score = corpus_bleu(references_tokenized_corpus, predictions_tokenized_corpus, 
                                    weights=weights, smoothing_function=smoothing)
    
    # Add overall BLEU score to data
    data['corpus_bleu'] = round(corpus_bleu_score * 100, 2)

def calculate_chrf(data):
    """Calculate chrF++ scores for all translations"""
    # chrF++ settings: character n-gram order 6, word n-gram order 2, beta 2
    chrf_metric = CHRF(char_order=6, word_order=2, beta=2)
    
    all_chrf_scores = []
    
    for item in data['results']:
        # Reference and prediction
        reference = [item['reference']]
        prediction = item['prediction']
        
        # Calculate chrF++
        chrf_score = chrf_metric.corpus_score([prediction], [reference]).score
        item['chrf'] = round(chrf_score, 2)
        all_chrf_scores.append(chrf_score)
    
    # Calculate average chrF++ score
    data['avg_chrf'] = round(np.mean(all_chrf_scores), 2)

def calculate_xcomet(data):
    """Calculate XCOMET scores for translations"""
    if not COMET_AVAILABLE:
        return
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Try to download and load XCOMET-XL model
    print("Downloading and loading XCOMET-XL model...")
    try:
        model_path = download_model("Unbabel/XCOMET-XL")
        model = load_from_checkpoint(model_path)
        print("Successfully loaded XCOMET-XL model")
    except Exception as e:
        print(f"Error loading XCOMET-XL model: {e}")
        # Fall back to standard COMET model if XCOMET fails
        print("Falling back to standard COMET model...")
        model_path = download_model("wmt20-comet-da")
        model = load_from_checkpoint(model_path)
        print("Successfully loaded standard COMET model")
    
    # Prepare data for COMET/XCOMET
    # Since we only have reference and prediction, use empty string for source
    xcomet_data = []
    
    for item in data['results']:
        xcomet_data.append({
            "src": "",  # Empty source as we don't have it
            "mt": item['prediction'],
            "ref": item['reference']
        })
    
    # Calculate scores in batches
    batch_size = 6  # Slightly reduced batch size for larger models
    
    # Process predictions
    process_xcomet_batch(model, xcomet_data, data, batch_size, device)

def process_xcomet_batch(model, batch_data, data, batch_size, device):
    """Helper function to process a batch of data with XCOMET model"""
    print(f"Calculating scores for {len(batch_data)} translations...")
    if not batch_data:
        return
    
    # Model API
    model_output = model.predict(batch_data, batch_size=batch_size, gpus=0 if device == "cpu" else 1)
    
    # Handle different return types from model.predict
    scores = None
    
    if hasattr(model_output, 'scores'):
        # Object style API (newer COMET)
        scores = model_output.scores
    elif isinstance(model_output, dict) and 'scores' in model_output:
        # Dictionary style API
        scores = model_output['scores']
    else:
        # Legacy style API
        scores = model_output
    
    # Add scores to data
    for i in range(len(scores)):
        if i < len(data['results']):
            score_value = scores[i]
            # Handle if score is a list
            if isinstance(score_value, list):
                score_value = score_value[0]
            data['results'][i]['xcomet'] = round(float(score_value), 4)
            
            # If error spans are available, add them too
            if hasattr(model_output, 'metadata') and hasattr(model_output.metadata, 'error_spans') and i < len(model_output.metadata.error_spans):
                data['results'][i]['error_spans'] = model_output.metadata.error_spans[i]
    
    # Store system-level score
    if hasattr(model_output, 'system_score'):
        # Object style API (newer COMET)
        data['system_xcomet'] = round(float(model_output.system_score), 4)
    else:
        # Calculate system score as mean of segment scores
        flat_scores = [s[0] if isinstance(s, list) else s for s in scores]
        data['system_xcomet'] = round(float(np.mean(flat_scores)), 4)

def calculate_summary(data):
    """Calculate summary statistics for all metrics"""
    # Count samples
    sample_count = len(data['results'])
    
    # Prepare summary dictionary
    summary = {
        'sample_count': sample_count,
        'metrics': {
            'bleu': data.get('corpus_bleu', 0),
            'chrf': data.get('avg_chrf', 0)
        }
    }
    
    # Add XCOMET scores if available
    if COMET_AVAILABLE:
        summary['metrics']['xcomet'] = data.get('system_xcomet', 0)
    
    return summary

def print_summary(summary):
    """Print a summary of all metrics to the console"""
    print("\n=== METRICS SUMMARY ===")
    print(f"Number of samples: {summary['sample_count']}")
    
    print("\nOverall Scores:")
    print(f"  BLEU: {summary['metrics']['bleu']:.2f}")
    print(f"  chrF++: {summary['metrics']['chrf']:.2f}")
    
    if 'xcomet' in summary['metrics']:
        print(f"  XCOMET: {summary['metrics']['xcomet']:.4f}")

if __name__ == "__main__":
    main() 
