from datasets import load_dataset
import pandas as pd
import sys
from datetime import datetime
from collections import defaultdict

# Create a class to write to both file and console
class TeeWriter:
    def __init__(self, file):
        self.file = file
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        
    def flush(self):
        self.file.flush()
        self.stdout.flush()

# Create output filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"analysis_output_{timestamp}.txt"

def read_training_data():
    """Read the target translation file and return all translations"""
    target_texts = []
    total_lines = 0
    
    # Read target translations
    with open("training_data_100k_glosslm_dedup/target.txt", 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                total_lines += 1
                target_texts.append(line)
    
    return target_texts, total_lines

# Use context managers to handle file and output redirection
with open(output_file, 'w') as f:
    # Redirect stdout to both file and console
    sys.stdout = TeeWriter(f)
    
    print("Loading datasets and analyzing fieldwork languages in target file...")
    
    # Load fieldwork dataset
    try:
        field = load_dataset("wav2gloss/fieldwork", split="test")
        print("Field dataset loaded successfully")
    except Exception as e:
        print(f"Error loading fieldwork dataset: {e}")
        exit(1)

    # Read fieldwork ISO codes to map glottocodes to language names
    glottocode_to_name = {}
    with open("fieldwork_iso.txt", "r") as iso_f:
        lines = iso_f.readlines()
        # Skip header (first 2 lines)
        for line in lines[2:]:
            parts = [part.strip() for part in line.strip().split("|")]
            if len(parts) >= 3:
                glottocode = parts[1].strip()
                language_name = parts[2].strip()
                glottocode_to_name[glottocode] = language_name

    # Load training data
    print("Loading target translation file...")
    target_texts, total_lines = read_training_data()
    print(f"Total lines in target file: {total_lines}")
    print(f"Unique translations in target file: {len(set(target_texts))}")

    # Create a set for faster lookup
    target_texts_set = set(target_texts)
    
    # Count fieldwork samples that appear in target file
    language_counts = defaultdict(int)
    matching_examples = []
    
    print("\nAnalyzing fieldwork samples in target file...")
    for example in field:
        lang = example["language"]
        translation = example["translation"]
        
        if translation in target_texts_set:
            language_counts[lang] += 1
            matching_examples.append((lang, translation))

    # Print results
    print(f"\nFieldwork languages found in target file:")
    print("=" * 60)
    
    total_matches = 0
    languages_with_matches = 0
    
    # Sort by count descending, then by language name
    for glottocode, count in sorted(
        language_counts.items(), 
        key=lambda x: (-x[1], x[0])
    ):
        language_name = glottocode_to_name.get(glottocode, "Unknown")
        print(f"{language_name} ({glottocode}): {count} samples")
        total_matches += count
        languages_with_matches += 1

    print(f"\nSummary:")
    print(f"Total fieldwork languages: {len(set(field['language']))}")
    print(f"Fieldwork languages with samples in target file: {languages_with_matches}")
    print(f"Total fieldwork samples found in target file: {total_matches}")
    print(f"Percentage of fieldwork samples in target file: {total_matches/len(field)*100:.2f}%")

    # Show some example matches
    if matching_examples:
        print(f"\nExample matches (first 10):")
        for lang, translation in matching_examples[:10]:
            language_name = glottocode_to_name.get(lang, "Unknown")
            print(f"  {language_name} ({lang}): {translation}")

# Restore original stdout
sys.stdout = sys.__stdout__
print(f"\nAnalysis results have been saved to: {output_file}")
