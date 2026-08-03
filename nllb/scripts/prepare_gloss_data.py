#!/usr/bin/env python
"""
Prepare gloss-to-translation data for NLLB fine-tuning.
Converts raw gloss.txt and translation.txt into JSONL format with language codes.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GlossDataPreparator:
    """Prepares gloss-to-translation data for NLLB fine-tuning."""
    
    def __init__(self, gloss_file: str, translation_file: str, output_dir: str, 
                 use_single_lang_tag: bool = True, create_test_split: bool = True):
        """
        Initialize data preparator.
        
        Args:
            gloss_file: Path to gloss.txt file (source sentences with language tags)
            translation_file: Path to translation.txt file (target English translations)
            output_dir: Directory to save processed datasets
            use_single_lang_tag: If True, use single 'gls_Latn' tag for all glosses.
                                 If False, use language-specific tags like 'gloss_nor', 'gloss_krl'.
                                 Recommended: True (better for low-resource languages)
            create_test_split: If True, create train/val/test splits (0.8/0.1/0.1).
                               If False, create train/val splits only (0.9/0.1).
        """
        self.gloss_file = Path(gloss_file)
        self.translation_file = Path(translation_file)
        self.output_dir = Path(output_dir)
        self.use_single_lang_tag = use_single_lang_tag
        self.create_test_split = create_test_split
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_language_tag_and_gloss(self, gloss_line: str) -> Tuple[str, str]:
        """
        Extract language tag and gloss from a line like 'nor word1 word2 ...'
        
        Args:
            gloss_line: Line from gloss.txt
            
        Returns:
            Tuple of (language_tag, gloss_text)
        """
        parts = gloss_line.strip().split(maxsplit=1)
        if len(parts) == 2:
            return parts[0], parts[1]
        elif len(parts) == 1:
            # Handle lines with only language tag (edge case)
            return parts[0], ""
        else:
            return "unk", ""
    
    def read_parallel_data(self) -> Tuple[List[Tuple[str, str]], List[str]]:
        """
        Read parallel gloss and translation files.
        
        Returns:
            Tuple of (list of (lang_tag, gloss) tuples, list of translations)
        """
        logger.info(f"Reading gloss file: {self.gloss_file}")
        with open(self.gloss_file, 'r', encoding='utf-8') as f:
            gloss_lines = f.readlines()
        
        logger.info(f"Reading translation file: {self.translation_file}")
        with open(self.translation_file, 'r', encoding='utf-8') as f:
            translation_lines = f.readlines()
        
        assert len(gloss_lines) == len(translation_lines), \
            f"Mismatched line counts: {len(gloss_lines)} glosses vs {len(translation_lines)} translations"
        
        logger.info(f"Total parallel sentences: {len(gloss_lines)}")
        
        # Extract language tags and glosses
        gloss_data = []
        for line in gloss_lines:
            lang_tag, gloss_text = self.extract_language_tag_and_gloss(line)
            gloss_data.append((lang_tag, gloss_text))
        
        translations = [line.strip() for line in translation_lines]
        
        return gloss_data, translations
    
    def analyze_languages(self, gloss_data: List[Tuple[str, str]]) -> Dict[str, int]:
        """
        Analyze language distribution in the dataset.
        
        Args:
            gloss_data: List of (lang_tag, gloss) tuples
            
        Returns:
            Dictionary of language counts
        """
        lang_counts = Counter([lang_tag for lang_tag, _ in gloss_data])
        
        logger.info(f"\nLanguage distribution ({len(lang_counts)} languages):")
        for lang, count in lang_counts.most_common():
            logger.info(f"  {lang}: {count} sentences")
        
        return dict(lang_counts)
    
    def create_train_val_split(self, data: List[Dict], 
                                val_ratio: float = 0.1,
                                test_ratio: float = 0.0,
                                seed: int = 42):
        """
        Split data into train and validation sets with stratification by language.
        
        Args:
            data: List of data dictionaries
            val_ratio: Ratio of validation data (default: 0.1 for 10%)
            test_ratio: Ratio of test data (0.0 for no test split)
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, val_data) or (train_data, val_data, test_data)
        """
        random.seed(seed)
        
        # Group by source language for stratified split
        lang_groups = {}
        for item in data:
            src_lang = list(item['translation'].keys())[0]  # First key is source lang
            if src_lang not in lang_groups:
                lang_groups[src_lang] = []
            lang_groups[src_lang].append(item)
        
        train_data = []
        val_data = []
        test_data = []
        
        # Stratified split per language
        for lang, items in lang_groups.items():
            random.shuffle(items)
            
            if test_ratio > 0:
                # 3-way split: train/val/test
                test_size = max(1, int(len(items) * test_ratio))
                val_size = max(1, int(len(items) * val_ratio))
                
                test_data.extend(items[:test_size])
                val_data.extend(items[test_size:test_size + val_size])
                train_data.extend(items[test_size + val_size:])
            else:
                # 2-way split: train/val
                val_size = max(1, int(len(items) * val_ratio))
                val_data.extend(items[:val_size])
                train_data.extend(items[val_size:])
        
        # Shuffle final splits
        random.shuffle(train_data)
        random.shuffle(val_data)
        if test_data:
            random.shuffle(test_data)
        
        logger.info(f"\nData split:")
        logger.info(f"  Train: {len(train_data)} examples ({len(train_data)/len(data)*100:.1f}%)")
        logger.info(f"  Validation: {len(val_data)} examples ({len(val_data)/len(data)*100:.1f}%)")
        if test_data:
            logger.info(f"  Test: {len(test_data)} examples ({len(test_data)/len(data)*100:.1f}%)")
        
        if test_ratio > 0:
            return train_data, val_data, test_data
        else:
            return train_data, val_data
    
    def create_train_val_split_single_tag(self, data: List[Dict], 
                                           val_ratio: float = 0.1,
                                           test_ratio: float = 0.0,
                                           seed: int = 42):
        """
        Split data with stratification by original language (for single tag mode).
        
        Args:
            data: List of data dictionaries with 'source_language' field
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data (0.0 for no test split)
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, val_data) or (train_data, val_data, test_data)
        """
        random.seed(seed)
        
        # Group by original source language for stratified split
        lang_groups = {}
        for item in data:
            src_lang = item.get('source_language', 'unk')
            if src_lang not in lang_groups:
                lang_groups[src_lang] = []
            lang_groups[src_lang].append(item)
        
        train_data = []
        val_data = []
        test_data = []
        
        # Stratified split per language
        for lang, items in lang_groups.items():
            random.shuffle(items)
            
            if test_ratio > 0:
                # 3-way split: train/val/test
                test_size = max(1, int(len(items) * test_ratio))
                val_size = max(1, int(len(items) * val_ratio))
                
                test_data.extend(items[:test_size])
                val_data.extend(items[test_size:test_size + val_size])
                train_data.extend(items[test_size + val_size:])
            else:
                # 2-way split: train/val
                val_size = max(1, int(len(items) * val_ratio))
                val_data.extend(items[:val_size])
                train_data.extend(items[val_size:])
        
        # Remove source_language field before saving
        all_data = train_data + val_data + test_data
        for item in all_data:
            item.pop('source_language', None)
        
        # Shuffle final splits
        random.shuffle(train_data)
        random.shuffle(val_data)
        if test_data:
            random.shuffle(test_data)
        
        logger.info(f"\nData split (stratified by {len(lang_groups)} original languages):")
        logger.info(f"  Train: {len(train_data)} examples ({len(train_data)/len(data)*100:.1f}%)")
        logger.info(f"  Validation: {len(val_data)} examples ({len(val_data)/len(data)*100:.1f}%)")
        if test_data:
            logger.info(f"  Test: {len(test_data)} examples ({len(test_data)/len(data)*100:.1f}%)")
        
        if test_ratio > 0:
            return train_data, val_data, test_data
        else:
            return train_data, val_data
    
    def save_to_jsonl(self, data: List[Dict], output_file: Path):
        """
        Save data to JSONL format compatible with HuggingFace datasets.
        
        Args:
            data: List of data dictionaries
            output_file: Path to output JSONL file
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Saved {len(data)} examples to {output_file}")
    
    def prepare_dataset(self, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42):
        """
        Main method to prepare the complete dataset.
        
        Args:
            val_ratio: Ratio of validation data (default: 0.1)
            test_ratio: Ratio of test data (default: 0.1 if create_test_split=True, else 0.0)
            seed: Random seed for reproducibility
        """
        # Read parallel data
        gloss_data, translations = self.read_parallel_data()
        
        # Analyze language distribution
        lang_counts = self.analyze_languages(gloss_data)
        
        # Save language statistics
        stats_file = self.output_dir / 'language_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(lang_counts, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved language statistics to {stats_file}")
        
        # Create dataset in JSONL format
        dataset = []
        skipped = 0
        
        if self.use_single_lang_tag:
            logger.info("\n✓ Using SINGLE language tag: gls_Latn")
            logger.info("  Rationale: All glosses use English-based conventions (1SG, PST, NOM, etc.)")
            logger.info("  Benefits: Better for low-resource languages, more training data per tag")
            src_lang_code = "gls_Latn"
        else:
            logger.info("\n✓ Using LANGUAGE-SPECIFIC tags: gloss_nor, gloss_krl, etc.")
            logger.info("  Benefits: Can track per-language performance")
        
        for (lang_tag, gloss_text), translation in zip(gloss_data, translations):
            if not gloss_text or not translation:
                skipped += 1
                continue
            
            # Create language code
            if self.use_single_lang_tag:
                src_lang_code = "gls_Latn"  # Single tag for all glosses
            else:
                src_lang_code = f"gloss_{lang_tag}"  # Language-specific tag
            
            tgt_lang_code = "eng_Latn"  # NLLB code for English
            
            entry = {
                "translation": {
                    src_lang_code: gloss_text,
                    tgt_lang_code: translation
                }
            }
            # Store original language for stratification even with single tag
            if self.use_single_lang_tag:
                entry["source_language"] = lang_tag
            
            dataset.append(entry)
        
        logger.info(f"\nProcessed {len(dataset)} valid examples (skipped {skipped} empty entries)")
        
        # Determine test ratio
        actual_test_ratio = test_ratio if self.create_test_split else 0.0
        
        # Split into train/val/test (stratified by original language if using single tag)
        if self.use_single_lang_tag:
            # Stratify by original language but use single tag
            logger.info("Stratifying split by original language for balanced representation")
            split_result = self.create_train_val_split_single_tag(dataset, val_ratio, actual_test_ratio, seed)
        else:
            split_result = self.create_train_val_split(dataset, val_ratio, actual_test_ratio, seed)
        
        # Unpack results
        if self.create_test_split:
            train_data, val_data, test_data = split_result
        else:
            train_data, val_data = split_result
            test_data = None
        
        # Save to JSONL files
        self.save_to_jsonl(train_data, self.output_dir / 'train.jsonl')
        self.save_to_jsonl(val_data, self.output_dir / 'val.jsonl')
        if test_data:
            self.save_to_jsonl(test_data, self.output_dir / 'test.jsonl')
        
        # Save language codes for model extension
        lang_codes_file = self.output_dir / 'language_codes.txt'
        with open(lang_codes_file, 'w', encoding='utf-8') as f:
            if self.use_single_lang_tag:
                f.write("gls_Latn\n")
                logger.info(f"Saved 1 language code to {lang_codes_file}: gls_Latn")
            else:
                unique_langs = sorted(set([lang_tag for lang_tag, _ in gloss_data]))
                for lang in unique_langs:
                    f.write(f"gloss_{lang}\n")
                logger.info(f"Saved {len(unique_langs)} unique language codes to {lang_codes_file}")
        
        logger.info("\n✓ Dataset preparation completed successfully!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare gloss-to-translation data for NLLB fine-tuning')
    parser.add_argument('--gloss_file', type=str, required=True, 
                        help='Path to gloss.txt file')
    parser.add_argument('--translation_file', type=str, required=True,
                        help='Path to translation.txt file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed datasets')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio (default: 0.1 for 10%%)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Test set ratio (default: 0.1 for 10%%, matches LSTM setup)')
    parser.add_argument('--no_test_split', action='store_true',
                        help='Skip test split, only create train/val (0.9/0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--use_single_lang_tag', type=bool, default=True,
                        help='Use single gls_Latn tag instead of language-specific tags (default: True, RECOMMENDED for low-resource languages)')
    parser.add_argument('--language_specific_tags', action='store_true',
                        help='Use language-specific tags (gloss_nor, gloss_krl, etc.) instead of single tag')
    
    args = parser.parse_args()
    
    # Determine which tagging approach to use
    use_single = args.use_single_lang_tag and not args.language_specific_tags
    
    # Determine if test split should be created
    create_test = not args.no_test_split
    
    preparator = GlossDataPreparator(
        gloss_file=args.gloss_file,
        translation_file=args.translation_file,
        output_dir=args.output_dir,
        use_single_lang_tag=use_single,
        create_test_split=create_test
    )
    
    preparator.prepare_dataset(val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)


if __name__ == "__main__":
    main()
