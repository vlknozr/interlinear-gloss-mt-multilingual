#!/usr/bin/env python
"""
Extend NLLB model with new language tokens for gloss languages.
Adapted from PARME repository: https://github.com/DOLMA-NLP/PARME
"""

import json
import shutil
import torch
import logging
from pathlib import Path
from typing import Dict, List
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, M2M100ForConditionalGeneration, AutoConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extend_tokenizer_and_model(model, tokenizer, new_lang_code: str, similar_lang_code: str):
    """
    Extend NLLB tokenizer and model with new language tag.
    
    Args:
        model: NLLB model
        tokenizer: NLLB tokenizer
        new_lang_code: New language code to add (e.g., "gloss_nor")
        similar_lang_code: Existing similar language code for initialization (e.g., "eng_Latn")
    """
    # Check if the new language code already exists
    if new_lang_code not in tokenizer.get_vocab():
        logger.info(f"Adding new language token: {new_lang_code}")
        
        # Add new token to tokenizer as a special token
        num_added_tokens = tokenizer.add_tokens([new_lang_code], special_tokens=True)
        
        # Resize token embeddings to accommodate new token
        model.resize_token_embeddings(len(tokenizer))
        
        # Get embedding of similar language to initialize new language
        similar_token_id = tokenizer.convert_tokens_to_ids(similar_lang_code)
        
        if similar_token_id == tokenizer.unk_token_id:
            logger.warning(f"Similar language '{similar_lang_code}' not found, using random initialization")
            # Random initialization will be used by default
        else:
            # Copy embedding from similar language
            similar_lang_embedding = model.get_input_embeddings().weight[similar_token_id].clone()
            
            # Initialize new language embedding with similar language
            new_token_id = tokenizer.convert_tokens_to_ids(new_lang_code)
            with torch.no_grad():
                model.get_input_embeddings().weight[new_token_id] = similar_lang_embedding
            
            logger.info(f"✓ Initialized {new_lang_code} with {similar_lang_code} embedding")
    else:
        logger.info(f"Language {new_lang_code} already exists in tokenizer")


def prepare_nllb_model(model_name: str, new_langs: List[str], similar_lang: str):
    """
    Prepare NLLB model with new language tags.
    
    Args:
        model_name: HuggingFace model identifier (e.g., "facebook/nllb-200-distilled-600M")
        new_langs: List of new language codes to add
        similar_lang: Similar language code for initialization
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model and tokenizer from {model_name}")
    
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name, config=config)
    
    logger.info(f"Original vocabulary size: {len(tokenizer)}")
    logger.info(f"Original model embeddings: {model.model.shared.weight.shape[0]}")
    
    # Add new languages
    for new_lang in new_langs:
        extend_tokenizer_and_model(model, tokenizer, new_lang, similar_lang)
    
    logger.info(f"Extended vocabulary size: {len(tokenizer)}")
    logger.info(f"Extended model embeddings: {model.model.shared.weight.shape[0]}")
    
    return model, tokenizer


def load_language_codes(lang_codes_file: str) -> List[str]:
    """
    Load language codes from file.
    
    Args:
        lang_codes_file: Path to file containing language codes (one per line)
        
    Returns:
        List of language codes
    """
    with open(lang_codes_file, 'r', encoding='utf-8') as f:
        lang_codes = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(lang_codes)} language codes from {lang_codes_file}")
    return lang_codes


def fix_saved_tokenizer_config(output_dir: Path, base_model_name: str) -> None:
    """
    Ensure tokenizer files needed for reload exist in output_dir.

    Newer transformers versions may omit sentencepiece.bpe.model and write
    `"vocab_file": null`, which breaks NllbTokenizer reload.
    """
    config_path = output_dir / "tokenizer_config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        changed = False
        if config.get("vocab_file") is None:
            config.pop("vocab_file", None)
            changed = True

        if isinstance(config.get("extra_special_tokens"), list):
            config["extra_special_tokens"] = {}
            changed = True

        if changed:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                f.write("\n")
            logger.info("Fixed tokenizer_config.json for transformers v4 compatibility")

    sp_model = output_dir / "sentencepiece.bpe.model"
    if not sp_model.exists():
        source = hf_hub_download(base_model_name, "sentencepiece.bpe.model")
        shutil.copy2(source, sp_model)
        logger.info("Copied sentencepiece.bpe.model from %s", base_model_name)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extend NLLB model with new gloss language tokens')
    parser.add_argument('--model_name', type=str, default='facebook/nllb-200-distilled-600M',
                        help='HuggingFace model identifier')
    parser.add_argument('--lang_codes_file', type=str, required=True,
                        help='File containing new language codes (one per line)')
    parser.add_argument('--similar_lang', type=str, default='eng_Latn',
                        help='Similar language code for initialization (default: eng_Latn)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory to save extended model and tokenizer')
    
    args = parser.parse_args()
    
    # Load language codes
    new_langs = load_language_codes(args.lang_codes_file)
    
    # Prepare extended model
    logger.info(f"\nExtending NLLB model with {len(new_langs)} new language codes...")
    model, tokenizer = prepare_nllb_model(
        model_name=args.model_name,
        new_langs=new_langs,
        similar_lang=args.similar_lang
    )
    
    # Save extended model and tokenizer
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nSaving extended model and tokenizer to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    fix_saved_tokenizer_config(output_dir, args.model_name)
    
    # Save metadata
    metadata = {
        'base_model': args.model_name,
        'new_languages': new_langs,
        'similar_language': args.similar_lang,
        'vocab_size': len(tokenizer),
        'num_new_tokens': len(new_langs)
    }
    
    with open(output_dir / 'extension_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("\n✓ Model extension completed successfully!")
    logger.info(f"  - Original vocab size: {len(tokenizer) - len(new_langs)}")
    logger.info(f"  - Extended vocab size: {len(tokenizer)}")
    logger.info(f"  - Added {len(new_langs)} new language tokens")


if __name__ == "__main__":
    main()
