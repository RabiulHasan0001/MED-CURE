import os, csv
from typing import List, Tuple, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

sapbert = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

def load_kg(csv_path: str) -> Optional[Dict]:
    if not os.path.exists(csv_path):
        return None
    triples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = [h.strip().lower() for h in header]
        
        def _get(row, key):
            alt = {"head": "subject", "relation": "predicate", "tail": "object"}
            idx = cols.index(key) if key in cols else cols.index(alt.get(key, ""))
            return row[idx].strip() if idx >= 0 else ""

        for row in reader:
            if len(row) < 3: continue
            h = _get(row, "head")
            r = _get(row, "relation") or "associated_with"  
            t = _get(row, "tail")
            if h and t:
                triples.append((h, r, t))

    by_cui: Dict[str, List[Tuple[str, str, str]]] = {}
    for h, r, t in triples:
        for node in (h, t):
            key = node.lower()  
            by_cui.setdefault(key, []).append((h, r, t))
    
    return {"triples": triples, "by_cui": by_cui}

def build_input_conditioned_kg(kg_index, entity_embs: np.ndarray, limit: int = 15):
    if not kg_index:
        return []
    
    # Link to CUIs via SapBERT sim
    kg_embs = sapbert.encode(list(kg_index["by_cui"].keys()))
    sims = np.dot(entity_embs, kg_embs.T)
    top_idx = np.argsort(-np.max(sims, axis=0))[:limit]
    
    subgraph = []
    seen = set()
    for idx in top_idx:
        cui = list(kg_index["by_cui"].keys())[idx]
        for tr in kg_index["by_cui"].get(cui, []):
            if tr not in seen:
                # Induce salient relations (filter to finding_of, treats, etc.)
                if tr[1] in {"finding_of", "treats", "contraindicated_with", "associated_with"}:
                    subgraph.append(tr)
                    seen.add(tr)
                    if len(subgraph) >= limit:
                        return subgraph
    return subgraph

def format_triples_for_prompt(triples: List[Tuple[str, str, str]]) -> str:
    if not triples:
        return "(no KG triples)"
    return "\n".join([f"• ({h}) -[{r}]-> ({t})" for h, r, t in triples])