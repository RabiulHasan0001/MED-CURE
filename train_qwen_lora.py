#!/usr/bin/env python3
import os, argparse, time, pathlib, gc, json, re, random, inspect
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch

# --------- Puhti-friendly defaults ---------
BASE = "/scratch/project_2014607"
HF_CACHE = f"{BASE}/hf_cache"
os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"] = HF_CACHE
os.environ["MPLCONFIGDIR"] = f"{BASE}/mpl_config"
os.environ["NLTK_DATA"] = f"{BASE}/nltk_data"
os.environ["WANDB_DISABLED"] = "true"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
import evaluate
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer,
    TrainerCallback, EarlyStoppingCallback,
    GenerationConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from sentence_transformers import SentenceTransformer
import spacy  # For entity extraction and qualifiers
from splade.models.transformer_rep import Splade  # Sparse retrieval (pip install splade)

# SAFE-CUI via SapBERT
from sapcuis_sapbert import safe_cui_score as safe_cui

# KG utils (updated for input-conditioned)
from kg_utils import load_kg, build_input_conditioned_kg, format_triples_for_prompt

# ----------------- Shared state (no globals) -----------------
EVAL_REF = {"inputs": [], "targets": []}

HIST = {
    "train_loss": [],
    "eval_rougeL": [],
    "eval_bertscore_f1": [],
    "eval_safe_cui": [],
}

# ---------- Readability (FKGL) ----------
def _syllables(word: str) -> int:
    word = word.lower()
    m = re.findall(r"[aeiouy]+", word)
    return max(1, len(m))

def flesch_kincaid_grade(text: str) -> float:
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]
    words = re.findall(r"\w+", text)
    if not sentences or not words:
        return 0.0
    syllables = sum(_syllables(w) for w in words)
    return 0.39 * (len(words)/len(sentences)) + 11.8 * (syllables/len(words)) - 15.59

# ---------- Prompt template (section-aware) ----------
SECTION_TEMPLATE = (
    "Write a concise **Discharge Summary** with EXACTLY these section headers:\n"
    "Diagnoses;\n"
    "Procedures/Interventions;\n"
    "Key Results & Status;\n"
    "Medications (with doses);\n"
    "Follow-up & Monitoring.\n"
    "Rules:\n"
    "- ONLY include facts present in the note (and KG context if provided).\n"
    "- DO NOT invent vitals, dates, or patient-education boilerplate.\n"
    "- DO NOT write generic advice like 'take medications as prescribed'.\n"
    "- Avoid duplicated phrases; be specific and compact.\n"
    "- Use short bullet points under each section; no long prose paragraphs.\n"
    "- Preserve temporal order: indication before intervention, treatment before response.\n"
)

# Section-aware priors (weight higher for key sections)
SECTION_PRIORS = {
    "diagnoses": 1.5,
    "procedures": 1.5,
    "medications": 1.2,
    "follow-up": 1.2,
    # Add more as needed
}

def parse_args():
    p = argparse.ArgumentParser("Qwen2.5-7B QLoRA fine-tuning on MIMIC/Open-i (+RAG/KG)")
    p.add_argument("--mimic_csv", type=str, default=f"{BASE}/mimic.csv")
    p.add_argument("--openi_path", type=str, default=None)  # Optional Open-i dataset path
    p.add_argument("--checkpoint", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--output_root", type=str, default=f"{BASE}/models")
    p.add_argument("--results_root", type=str, default=f"{BASE}/results")

    # training/runtime
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--train_bs", type=int, default=2)
    p.add_argument("--eval_bs", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--max_src_len", type=int, default=512)
    p.add_argument("--max_tgt_len", type=int, default=220)
    p.add_argument("--token_budget", type=int, default=1024)  # Fixed token budget for evidence
    p.add_argument("--subset", type=int, default=0)
    p.add_argument("--eval_subset", type=int, default=200)
    p.add_argument("--save_merged", action="store_true")
    p.add_argument("--resume_from", type=str, default="")

    # evaluation printing
    p.add_argument("--preds_n", type=int, default=5)

    # generation
    p.add_argument("--gen_n", type=int, default=50)
    p.add_argument("--gen_all", action="store_true")
    p.add_argument("--gen_beams", type=int, default=8)
    p.add_argument("--use_zero_shot", action="store_true")  # For comparison

    # KG / retrieval paths
    p.add_argument("--kg_csv", type=str, default=f"{BASE}/kg/clin_kg.csv")  # UMLS triples

    return p.parse_args()

def load_data(args):
    # Load MIMIC
    df_mimic = pd.read_csv(args.mimic_csv)
    if not {"input", "target"}.issubset(set(df_mimic.columns)):
        raise ValueError("MIMIC CSV must contain columns: input, target")
    df_mimic = df_mimic[["input", "target"]].dropna()
    if args.subset and args.subset < len(df_mimic):
        df_mimic = df_mimic.sample(n=args.subset, random_state=42)
    tr_m, va_m = train_test_split(df_mimic, test_size=0.2, random_state=42)
    ds_tr_m = Dataset.from_pandas(tr_m)
    ds_va_m = Dataset.from_pandas(va_m)

    # Load Open-i if provided (Findings-to-Impression)
    ds_tr_o, ds_va_o = None, None
    if args.openi_path:
        ds_openi = load_dataset(args.openi_path)  # Assume HuggingFace path or local
        # Assume columns: 'findings' as input, 'impression' as target
        tr_o, va_o = train_test_split(ds_openi['train'].to_pandas(), test_size=0.2)
        ds_tr_o = Dataset.from_pandas(tr_o)
        ds_va_o = Dataset.from_pandas(va_o)

    # Combine if both
    if ds_tr_o:
        ds_tr = Dataset.from_pandas(pd.concat([tr_m, tr_o]))
        ds_va = Dataset.from_pandas(pd.concat([va_m, va_o]))
    else:
        ds_tr = ds_tr_m
        ds_va = ds_va_m

    return ds_tr, ds_va

def make_tokenizer_and_model(ckpt):
    tok = AutoTokenizer.from_pretrained(ckpt, cache_dir=HF_CACHE, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(ckpt, cache_dir=HF_CACHE, torch_dtype=dtype)

    # Memory helpers
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()

    base.config.pad_token_id = tok.pad_token_id
    base.config.eos_token_id = tok.eos_token_id
    base.generation_config.pad_token_id = tok.pad_token_id
    base.generation_config.eos_token_id = tok.eos_token_id

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none"
    )
    model = get_peft_model(base, lora_cfg)
    return tok, model

# Retrieval Channel: Sparse-Dense with MMR 
def retrieve_evidence(text: str, token_budget: int, topk: int = 20, lambda_mmr: float = 0.7):
    # Sparse: SPLADE-v2
    splade = Splade("naver/splade-cocondenser-ensembledistil")  # Clinically tuned if available
    splade.eval()
    sparse_vec = splade.encode_queries([text])

    
    # For real: Use FAISS or similar for sparse search
    candidates = []  # List of (section, evidence_text, score)
    # Mock: Retrieve topk candidates with section priors
    for cand in mock_corpus_search(sparse_vec, topk):  # Implement actual search
        section = extract_section(cand)  # Parse section from evidence
        prior = SECTION_PRIORS.get(section.lower(), 1.0)
        candidates.append((section, cand, score * prior))

    # Dense re-ranking: SapBERT
    sapbert = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    q_emb = sapbert.encode([text])
    c_embs = sapbert.encode([c[1] for c in candidates])
    scores = np.dot(q_emb, c_embs.T)[0]

    # MMR for diversity under token budget
    selected = []
    while candidates and sum(len(tok(c[1])) for c in selected) < token_budget:
        # MMR score = relevance - max_sim_to_selected
        mmr_scores = scores - lambda_mmr * max_sim_to_selected(c_embs, [sapbert.encode(s[1]) for s in selected])
        idx = np.argmax(mmr_scores)
        selected.append(candidates.pop(idx))
    return selected  # List of diverse, section-aware evidence

# Mock corpus search (replace with real index)
def mock_corpus_search(vec, topk):
    return ["mock_ev1", "mock_ev2"] * topk  # Placeholder

def extract_section(text: str) -> str:
    # Regex or NLP to extract section header
    return "diagnoses"  # Placeholder

# KG Retrieval (Input-Conditioned) 
nlp = spacy.load("en_core_sci_sm")  # Clinical NER

def kg_retrieve_and_constrain(kg_index, text: str, topk: int = 5):
    # Link spans to CUIs with SapBERT
    entities = [ent.text for ent in nlp(text).ents]  # Extract clinical entities
    sapbert = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    e_embs = sapbert.encode(entities)

    # Induce relations from KG
    subgraph = build_input_conditioned_kg(kg_index, e_embs, topk)  

    # Confidence-weighted entity masking (stabilize synonyms)
    masked_text = mask_entities(text, entities, confidence=0.8)  

    # Relation plausibility: Favor valid sequences
    plausible_triples = filter_plausible_relations(subgraph, text)  

    return format_triples_for_prompt(plausible_triples)

def mask_entities(text: str, entities: List, confidence: float):
    # Replace low-conf synonyms with CUI placeholders
    for e in entities:
        if random.random() < confidence:
            text = text.replace(e, f"[CUI:{hash(e)}]")  
    return text

def filter_plausible_relations(subgraph, text):
    # Minimal-revision: Discard unsupported triples
    # Temporal: Check order ('indication' before 'treats')
    return [tr for tr in subgraph if is_plausible(tr, text)]  # Implement check

def is_plausible(triple, text):
    # Check if relation supported in evidence or graph
    return True  

def build_tokenizer_fn(tok, max_src, max_tgt, args, kg_index):
    max_total = max_src + max_tgt
    def _fn(batch):
        convs = []
        for src in batch["input"]:
            # Retrieve evidence (RAG)
            evidence = retrieve_evidence(src, args.token_budget)

            # KG channel
            kg_prompt = kg_retrieve_and_constrain(kg_index, src)

            user_content = f"{SECTION_TEMPLATE}\n\nEvidence:\n{evidence}\n\nKG:\n{kg_prompt}\n\nSummarize:\n\n{src}"
            convs.append([
                {"role": "system", "content": "You are a clinical summarization assistant."},
                {"role": "user", "content": user_content}
            ])

        prompt_texts = tok.apply_chat_template(convs, add_generation_prompt=True, tokenize=False)
        targets = [str(t) for t in batch["target"]]
        full_texts = [p + t for p, t in zip(prompt_texts, targets)]

        prompt_enc = tok(prompt_texts, truncation=True, max_length=max_src, padding="max_length")
        full_enc = tok(full_texts, truncation=True, max_length=max_total, padding="max_length")

        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]

        labels = []
        for ids, p_am in zip(input_ids, prompt_enc["attention_mask"]):
            prompt_len = int(sum(p_am))
            lab = ids.copy()
            for i in range(prompt_len):
                lab[i] = -100
            labels.append(lab)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    return _fn

def build_metrics(tok):
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    def _metrics(eval_pred):
        preds, labels = eval_pred.predictions, eval_pred.label_ids
        decoded_preds = tok.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tok.batch_decode(labels, skip_special_tokens=True)

        rouge_res = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        bert_res = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")

        # SAFE-CUI
        safe_scores = []
        for p, r, s in zip(decoded_preds, decoded_labels, EVAL_REF["inputs"]):
            score, parts = safe_cui(p, r, s)
            safe_scores.append(score)
        avg_safe = np.mean(safe_scores)

        return {
            "rougeL": rouge_res["rougeL"],
            "bertscore_f1": np.mean(bert_res["f1"]),
            "safe_cui": avg_safe,
        }

    return _metrics

class RelationPlausibilityCallback(TrainerCallback):
    # During generation, enforce plausibility
    def on_generate(self, args, state, control, **kwargs):
        # Minimal-revision objective: Penalize unsupported triples
        pass  # Implement in generation config

def main():
    a = parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    STAMP = time.strftime("%Y%m%d-%H%M%S")
    MODEL_DIR = f"{a.output_root}/qwen-lora-{STAMP}"
    RESULTS_DIR = f"{a.results_root}/qwen-lora-{STAMP}"
    pathlib.Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))

    # Data
    train_ds, val_ds = load_data(a)
    full_val_ds = val_ds
    if a.eval_subset and a.eval_subset < len(val_ds):
        val_ds = val_ds.select(range(a.eval_subset))

    EVAL_REF["inputs"] = list(val_ds["input"])
    EVAL_REF["targets"] = list(val_ds["target"])

    print(f"Dataset: train={len(train_ds)} | val={len(val_ds)} (full={len(full_val_ds)})")

    tok, model = make_tokenizer_and_model(a.checkpoint)

    # Load KG (UMLS-grounded)
    kg_index = load_kg(a.kg_csv)

    tok_fn = build_tokenizer_fn(tok, a.max_src_len, a.max_tgt_len, a, kg_index)

    tok_train = train_ds.map(tok_fn, batched=True, remove_columns=train_ds.column_names)
    tok_val = val_ds.map(tok_fn, batched=True, remove_columns=val_ds.column_names)

    # LoRA params report
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable: {trainable:,} | all: {total:,} | %: {100*trainable/total:.4f}")

    collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model, padding="longest", label_pad_token_id=-100)
    metrics = build_metrics(tok)

    # TrainingArgs (adaptive)
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    kw = {
        "output_dir": MODEL_DIR,
        "learning_rate": a.lr,
        "per_device_train_batch_size": a.train_bs,
        "per_device_eval_batch_size": a.eval_bs,
        "gradient_accumulation_steps": a.grad_accum,
        "num_train_epochs": a.epochs,
        "weight_decay": 0.1,
        "max_grad_norm": 1.0,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "save_total_limit": 2,
        "fp16": torch.cuda.is_available(),
        "remove_unused_columns": False,
        "predict_with_generate": True,
        "generation_num_beams": 8,
        "metric_for_best_model": "rougeL",
        "load_best_model_at_end": True,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "report_to": "none",
    }
    if _supports_arg(sig, "gradient_checkpointing"):
        kw["gradient_checkpointing"] = True
    args = Seq2SeqTrainingArguments(**kw)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tok_train,
        eval_dataset=tok_val,
        data_collator=collator,
        compute_metrics=metrics,
        callbacks=[
            RelationPlausibilityCallback(),
            EarlyStoppingCallback(early_stopping_patience=2),
        ],
    )

    # Generation config (with constraints)
    gen_cfg = GenerationConfig(
        num_beams=a.gen_beams,
        do_sample=False,
        max_new_tokens=a.max_tgt_len,
        pad_token_id=tok.pad_token_id,
        # Add plausibility penalty
    )

    # Train
    trainer.train(resume_from_checkpoint=a.resume_from if a.resume_from else None)

    # Save adapter
    ADAPTER_DIR = f"{MODEL_DIR}/adapter"
    tok.save_pretrained(ADAPTER_DIR)
    trainer.model.save_pretrained(ADAPTER_DIR)
    print("LoRA adapter saved to:", ADAPTER_DIR)

    # Final metrics on full val
    tok_val_full = full_val_ds.map(tok_fn, batched=True, remove_columns=full_val_ds.column_names)
    EVAL_REF["inputs"] = list(full_val_ds["input"])
    EVAL_REF["targets"] = list(full_val_ds["target"])
    final_metrics = trainer.evaluate(eval_dataset=tok_val_full)
    with open(f"{RESULTS_DIR}/metrics_final.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    print("Final metrics saved to:", f"{RESULTS_DIR}/metrics_final.json")

    # Zero-shot comparison if flagged
    if a.use_zero_shot:
        # Implement zero/few-shot prompting without fine-tuning
        pass  # Add logic

   

if __name__ == "__main__":
    main()