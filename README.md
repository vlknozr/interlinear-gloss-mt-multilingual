# Multilingual Translation from Interlinear Glosses for Low-Resource Languages

## Directory Structure

### GlossLM Dataset Processing

#### [`glosslm_gloss_generation/`](glosslm_gloss_generation/)
Evaluation scripts for GlossLM model gloss generation.

- **Purpose**: Evaluate GlossLM model performance on test sets

#### [`glosslm_llm_finetune_data/`](glosslm_llm_finetune_data/)
Scripts for processing GlossLM dataset for LLM fine-tuning.

- **Purpose**: Download, filter, and prepare GlossLM data for model fine-tuning

#### [`shared_task_lang_glosslm/`](shared_task_lang_glosslm/)
Extraction scripts for out-of-domain languages from GlossLM corpus.

- **Purpose**: Extract data for Shared Task languages

### Language Model Fine-tuning

#### [`llm_finetune/`](llm_finetune/)
Fine-tuning Qwen models for gloss translation.

- **Purpose**: Fine-tune large language models to translate glosses to natural English

#### [`llm_retrieval/`](llm_retrieval/)
Retrieval-based approaches for gloss translation and dictionary building.

- **Purpose**: Implement retrieval mechanisms for few-shot learning

### Neural Machine Translation

#### [`nmt_training/`](nmt_training/)
Neural machine translation training with BPE tokenization using OpenNMT-py.

- **Purpose**: Train NMT models for gloss-to-translation and transcription-to-translation tasks

#### [`nllb/`](nllb/)
Fine-tuning NLLB-200 for gloss-to-translation.

- **Purpose**: Ffine-tune facebook/nllb-200-distilled-600M on multilingual gloss data

### Fieldwork Data Processing

#### [`wav2gloss_fieldwork/`](wav2gloss_fieldwork/)
Processing scripts for wav2gloss/fieldwork dataset.

- **Purpose**: Download, extract, and analyze fieldwork language data

