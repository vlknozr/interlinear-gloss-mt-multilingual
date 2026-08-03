#!/usr/bin/env python
"""Translate gloss lines to English using the base NLLB-200 model (no fine-tuning)."""

import argparse
import logging
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, M2M100ForConditionalGeneration

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"


def load_gloss_lines(input_file: Path) -> list[str]:
    with open(input_file, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def translate_batch(
    model,
    tokenizer,
    glosses: list[str],
    src_lang: str,
    tgt_lang: str,
    max_length: int,
    num_beams: int,
) -> list[str]:
    tokenizer.src_lang = src_lang
    inputs = tokenizer(
        glosses,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate gloss lines with facebook/nllb-200-distilled-600M"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input file with one gloss per line",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output file for translations only (one per line). Use '-' for stdout.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HuggingFace model id or local path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        default="eng_Latn",
        help="Source NLLB language code (default: eng_Latn; base model has no gls_Latn)",
    )
    parser.add_argument("--tgt_lang", type=str, default="eng_Latn")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda or cpu (default: cuda if available)",
    )
    args = parser.parse_args()

    input_file = Path(args.input_file)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    glosses = load_gloss_lines(input_file)
    logger.info("Loaded %d gloss lines from %s", len(glosses), input_file)
    logger.info("Loading model %s on %s", args.model_name, device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(args.model_name)
    model = model.to(device)
    model.eval()

    translations: list[str] = []
    for start in tqdm(range(0, len(glosses), args.batch_size), desc="Translating"):
        batch = glosses[start : start + args.batch_size]
        batch_translations = translate_batch(
            model,
            tokenizer,
            batch,
            args.src_lang,
            args.tgt_lang,
            args.max_length,
            args.num_beams,
        )
        translations.extend(batch_translations)

    if len(translations) != len(glosses):
        raise RuntimeError(
            f"Translation count mismatch: {len(translations)} outputs for {len(glosses)} inputs"
        )

    output_text = "\n".join(translations) + ("\n" if translations else "")
    if args.output_file == "-":
        print(output_text, end="")
    else:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text, encoding="utf-8")
        logger.info("Wrote %d translations to %s", len(translations), out_path)


if __name__ == "__main__":
    main()
