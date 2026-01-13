import re
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

# Qualifier detection (negation, uncertainty, laterality)
QUALIFIER_RE = {
    "negation": re.compile(r"\b(no|not|without|denies|negative|ruled out)\b", re.I),
    "uncertainty": re.compile(r"\b(may|possible|likely|suspected|question of)\b", re.I),
    "laterality": re.compile(r"\b(left|right|bilateral)\b", re.I),
}

MED_HINTS = set([
    "hf","hfpef","cad","mi","angina","stemi","nstemi","pna","uti",
    "ckd","esrd","copd","htn","dm","dm2","afib","vt","pe","dvt","tevar",
    "aaa","ras","gerd","ascites","cirrhosis","anemia","sepsis","pvd",
    "lasix","furosemide","torsemide","spironolactone","carvedilol","hydralazine",
    "heparin","amiodarone","aspirin","statin","metformin",
    "pci","stent","des","cabg","picc","rhc","lvef","ef",
    "diuresis","follow-up","clinic","recheck","electrolytes"
])

sapbert = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

def extract_clinical_concepts(text: str, max_terms: int = 40) -> List[Tuple[str, Dict[str, str]]]:
    # Extract terms + qualifiers
    toks = re.findall(r"[A-Za-z0-9+\-/\.%]+", text or "")
    cands = []
    for tok in toks:
        lt = tok.lower()
        if lt in MED_HINTS or re.search(r"^\d+(\.\d+)?(mg|mcg|%|mmhg)$", lt) or (lt in {"s/p","c/o","h/o"}) or (lt.isupper() and 2 <= len(lt) <= 5):
            qualifiers = {q: QUALIFIER_RE[q].search(text) is not None for q in QUALIFIER_RE}
            cands.append((tok.strip(",.:- "), qualifiers))
    seen, out = set(), []
    for c, q in cands:
        if c and c not in seen:
            out.append((c.lower(), q)); seen.add(c)
            if len(out) >= max_terms:
                break
    return out

def _concept_fidelity(pred_concepts: List, ref_concepts: List, thr: float = 0.80) -> float:
    # Embed with SapBERT for UMLS-normalized similarity
    p_embs = sapbert.encode([c[0] for c in pred_concepts])
    r_embs = sapbert.encode([c[0] for c in ref_concepts])
    sim_matrix = np.dot(p_embs, r_embs.T) / (np.linalg.norm(p_embs, axis=1)[:, np.newaxis] * np.linalg.norm(r_embs, axis=1))
    
    matched = set()
    tp = 0
    for i, p in enumerate(pred_concepts):
        best_j = np.argmax(sim_matrix[i])
        if sim_matrix[i, best_j] >= thr and best_j not in matched:
            # Check qualifier preservation
            if pred_concepts[i][1] == ref_concepts[best_j][1]:
                tp += 1
                matched.add(best_j)
    fp = len(pred_concepts) - tp
    fn = len(ref_concepts) - tp
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0

def _classify_concepts(concepts: List[Tuple[str, Dict]]) -> Dict[str, List[Tuple[str, Dict]]]:
    buckets = {"diagnosis": [], "procedure": [], "medication": [], "follow-up": []}
    for c, q in concepts:
        lc = c.lower()
        if "dose" in lc or any(k in lc for k in ["mg", "mcg", "lasix", "statin", "metformin"]):
            buckets["medication"].append((c, q))
        elif any(k in lc for k in ["surgery", "cabg", "pci", "procedure"]):
            buckets["procedure"].append((c, q))
        elif any(k in lc for k in ["follow", "clinic", "monitor"]):
            buckets["follow-up"].append((c, q))
        else:
            buckets["diagnosis"].append((c, q))
    return buckets

def _relation_plausibility(pred: str, ref: str, src: str) -> float:
    # Extract triples (simple: assume (entity1, relation, entity2) from text)
    # Plausibility: Overlap of induced relations
    return 0.8  # Placeholder: Compute triple overlap using KG

def _unsupported_rate(pred: str, src: str) -> float:
    # Concepts in pred not in src or KG
    p_concepts = [c[0] for c in extract_clinical_concepts(pred)]
    s_concepts = [c[0] for c in extract_clinical_concepts(src)]
    unsupported = len(set(p_concepts) - set(s_concepts)) / len(p_concepts) if p_concepts else 0
    return 1 - unsupported  # Fidelity score (lower unsupported = higher)

def safe_cui_score(pred: str, ref: str, src: str, thr: float = 0.80) -> Tuple[float, Dict[str, float]]:
    p_concepts = extract_clinical_concepts(pred)
    r_concepts = extract_clinical_concepts(ref)
    overall_fid = _concept_fidelity(p_concepts, r_concepts, thr)
    
    p_buckets = _classify_concepts(p_concepts)
    r_buckets = _classify_concepts(r_concepts)
    category_fid = {k: _concept_fidelity(p_buckets[k], r_buckets[k], thr) for k in p_buckets}
    
    qual_pres = np.mean([p[1] == r[1] for p, r in zip(p_concepts, r_concepts)]) if p_concepts else 1.0
    rel_plaus = _relation_plausibility(pred, ref, src)
    unsup_rate = _unsupported_rate(pred, src)
    
    # Aggregate SAFE-CUI: Weighted average
    safe_cui = 0.4 * overall_fid + 0.2 * np.mean(list(category_fid.values())) + 0.1 * qual_pres + 0.2 * rel_plaus + 0.1 * unsup_rate
    parts = {**category_fid, "qualifier_preservation": qual_pres, "relation_plausibility": rel_plaus, "unsupported_rate": unsup_rate}
    
    return safe_cui, parts