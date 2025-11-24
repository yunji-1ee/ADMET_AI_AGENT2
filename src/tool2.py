# 개선된 코드 -> 11.22
# === 개선안 ===
# Procedure Step에서 Modified Steps용 별도 dict 생성
# Mannitol integrity test가 포함된 chunk가 있으면 명시적으로 Modified Steps에 포함
# Section별 일반 top-k와 Modified Steps를 함께 반환하도록 함수 구조 개선

from typing import List
import os, json, faiss, numpy as np, re
from pathlib import Path
from sentence_transformers import SentenceTransformer

# --- 경로 설정 ---
IDXDIR  = Path(__file__).resolve().parent / ".." / "data" / "index"
DATADIR = Path(__file__).resolve().parent / ".." / "data" / "processed"

META  = json.load(open(IDXDIR / "meta.json", encoding="utf-8"))
INDEX = faiss.read_index(str(IDXDIR / "faiss.index"))

MODEL = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")

# --- chunk_id -> text (merged chunks 기반) ---
CHUNK_TEXT = {}
for line in open(DATADIR / "chunks_merged.jsonl", encoding="utf-8"):
    row = json.loads(line)
    cid = row.get("chunk_id") or f"C{hash(row.get('content')) % 100000}"
    CHUNK_TEXT[cid] = row.get("content", "")

# ----- keyword / section boost -----
KEY_BONUS = [
    (["mannitol", "paracellular", "cm s", "10^-", "10−", "×10"], 0.08),
    (["TER", "TEER", "ohm", "Ω", "Endohm", "electrode", "integrity"], 0.08),
    (["passive", "mM", "saturate", "saturation"], 0.05),
    (["efflux ratio", "uptake ratio", "1.5", "2.0"], 0.04),
    (["DMSO", "1%", "0.5%"], 0.04),
    (["solubility", "dissolution", "precipitation", "low solubility", "poorly soluble", "solubilizing agent"], 0.07),
    (["lipophilicity", "lipophilic", "logP", "logD", "BSA", "albumin", "non-specific binding", "adsorption", "mass balance"], 0.07),
    (["toxicity", "toxic", "compromised", "integrity check", "cell viability", "safety"], 0.07),
    (["molecular weight", "MW", "high MW", "large molecule"], 0.03),
    (["critical step", "caution", "important consideration"], 0.04),
]

# 특이 물성 property bonus
SPECIAL_PROPERTY_BONUS = {
    "low_solubility": 0.07,
    "high_lipophilicity": 0.07,
    "toxicity": 0.07,
    "active_transport": 0.05,
    "monolayer_integrity": 0.05,
    "co_solvent_effect": 0.05,
    "mass_balance": 0.04,
}

def _keyword_bonus(txt: str) -> float:
    t = (txt or "").lower()
    bonus = 0.0
    for toks, w in KEY_BONUS:
        if any(tok.lower() in t for tok in toks):
            bonus += w
    if re.search(r'(?:×|x)?\s*10\s*(?:[-−^])\s*\d+', t):
        bonus += 0.03
    return bonus

def _priority_boost(scores, ids):
    scores = scores.copy()
    for i, idx in enumerate(ids):
        m   = META[idx]
        sec = (m.get("section") or "").upper()
        if m.get("regulatory_flag") == "normative":
            scores[i] += 0.05
        if sec.startswith("BOX") or sec.startswith("PROCEDURE"):
            scores[i] += 0.02
        subsec = (m.get("subsection") or "").upper()
        if subsec and re.match(r'\d+', subsec):
            scores[i] += 0.01

        body = m.get("content") or CHUNK_TEXT.get(m.get("chunk_id"), "")
        scores[i] += _keyword_bonus(body)

        # 특이 물성 기반 bonus 추가
        for prop in m.get("pertains_to_property", []):
            scores[i] += SPECIAL_PROPERTY_BONUS.get(prop, 0)

    order = np.argsort(-scores)
    return scores[order], ids[order]

def _search_one(query: str, k: int, overfetch: int = 6):
    q = MODEL.encode([query], normalize_embeddings=True)
    scores, I = INDEX.search(q, k * overfetch)
    scores, I = scores[0], I[0]
    scores, I = _priority_boost(scores, I)
    return scores[:k], I[:k]

# ----- 핵심: Procedure Step 1~20 + section별 top-k -----
def retrieve_and_cite_structured(query: str, k_section: int = 2) -> dict:
    """
    RAG 검색 결과를 Procedure Step, Section top-k, Modified Steps로 구조화
    반환 형식: {"procedure_steps": [...], "section_topk": [...], "modified_steps": [...]}
    """
    scores, ids = _search_one(query, k=100)
    procedure_steps = {}
    section_topk = {}
    modified_steps = []
    seen_texts = set()

    for s, idx in zip(scores, ids):
        m = META[int(idx)]
        cid = m.get("chunk_id")
        sec = (m.get("section") or "").upper()
        subsec = m.get("subsection")
        txt = (m.get("content") or CHUNK_TEXT.get(cid, "") or "").replace("\n", " ")

        if txt in seen_texts:
            continue
        seen_texts.add(txt)

        # Procedure Step 1~20
        step = None
        if sec.startswith("PROCEDURE") and subsec and subsec.isdigit():
            step = int(subsec)
            if 1 <= step <= 20:
                procedure_steps[step] = f"[{cid} | {sec} subsec {subsec} · score={s:.3f}] {txt[:500]}"
                # 특이 케이스 (low solubility, toxicity, DMSO/BSA 조정 등)
                if any(prop in m.get("pertains_to_property", []) for prop in ["low_solubility", "high_lipophilicity", "toxicity"]):
                    modified_steps.append(f"[{cid} | Modified Step · score={s:.3f}] {txt[:500]}")
                # Mannitol integrity test 포함 여부
                if "mannitol" in txt.lower() or "integrity test" in txt.lower():
                    modified_steps.append(f"[{cid} | Mannitol integrity test · score={s:.3f}] {txt[:500]}")
                continue

        # 나머지 section top-k
        if sec not in section_topk:
            section_topk[sec] = []
        if len(section_topk[sec]) < k_section:
            section_topk[sec].append(f"[{cid} | {sec} {f'subsec {subsec}' if subsec else ''} · score={s:.3f}] {txt[:500]}")

    # 정렬 후 리스트화
    proc_list = [procedure_steps[step] for step in range(1, 21) if step in procedure_steps]
    section_list = [chunk for sec_chunks in section_topk.values() for chunk in sec_chunks]

    return {
        "procedure_steps": proc_list,
        "section_topk": section_list,
        "modified_steps": modified_steps
    }


# --- 예시 ---
if __name__ == "__main__":
    query = "How to handle low solubility compounds and DMSO co-solvent?"
    results = retrieve_and_cite_structured(query)
    for r in results:
        print(r)



'''

# 개선된 코드 -> 11.22
# 특이 물성 케이스와 KEY_BONUS 들을 더 추가하면서 내용을 보충함. 
from typing import List
import os, json, faiss, numpy as np, re
from pathlib import Path
from sentence_transformers import SentenceTransformer

# --- 경로 설정 ---
IDXDIR  = Path(__file__).resolve().parent / ".." / "data" / "index"
DATADIR = Path(__file__).resolve().parent / ".." / "data" / "processed"

META  = json.load(open(IDXDIR / "meta.json", encoding="utf-8"))
INDEX = faiss.read_index(str(IDXDIR / "faiss.index"))

MODEL = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")

# --- chunk_id -> text (merged chunks 기반) ---
CHUNK_TEXT = {}
for line in open(DATADIR / "chunks_merged.jsonl", encoding="utf-8"):
    row = json.loads(line)
    cid = row.get("chunk_id") or f"C{hash(row.get('content')) % 100000}"
    CHUNK_TEXT[cid] = row.get("content", "")

# ----- keyword / section boost -----
KEY_BONUS = [
    (["mannitol", "paracellular", "cm s", "10^-", "10−", "×10"], 0.08),
    (["TER", "TEER", "ohm", "Ω", "Endohm", "electrode", "integrity"], 0.08),
    (["passive", "mM", "saturate", "saturation"], 0.05),
    (["efflux ratio", "uptake ratio", "1.5", "2.0"], 0.04),
    (["DMSO", "1%", "0.5%"], 0.04),
    (["solubility", "dissolution", "precipitation", "low solubility", "poorly soluble", "solubilizing agent"], 0.07),
    (["lipophilicity", "lipophilic", "logP", "logD", "BSA", "albumin", "non-specific binding", "adsorption", "mass balance"], 0.07),
    (["toxicity", "toxic", "compromised", "integrity check", "cell viability", "safety"], 0.07),
    (["molecular weight", "MW", "high MW", "large molecule"], 0.03),
    (["critical step", "caution", "important consideration"], 0.04),
]

# 특이 물성 property bonus
SPECIAL_PROPERTY_BONUS = {
    "low_solubility": 0.07,
    "high_lipophilicity": 0.07,
    "toxicity": 0.07,
    "active_transport": 0.05,
    "monolayer_integrity": 0.05,
    "co_solvent_effect": 0.05,
    "mass_balance": 0.04,
}

def _keyword_bonus(txt: str) -> float:
    t = (txt or "").lower()
    bonus = 0.0
    for toks, w in KEY_BONUS:
        if any(tok.lower() in t for tok in toks):
            bonus += w
    if re.search(r'(?:×|x)?\s*10\s*(?:[-−^])\s*\d+', t):
        bonus += 0.03
    return bonus

def _priority_boost(scores, ids):
    scores = scores.copy()
    for i, idx in enumerate(ids):
        m   = META[idx]
        sec = (m.get("section") or "").upper()
        if m.get("regulatory_flag") == "normative":
            scores[i] += 0.05
        if sec.startswith("BOX") or sec.startswith("PROCEDURE"):
            scores[i] += 0.02
        subsec = (m.get("subsection") or "").upper()
        if subsec and re.match(r'\d+', subsec):
            scores[i] += 0.01

        body = m.get("content") or CHUNK_TEXT.get(m.get("chunk_id"), "")
        scores[i] += _keyword_bonus(body)

        # 특이 물성 기반 bonus 추가
        for prop in m.get("pertains_to_property", []):
            scores[i] += SPECIAL_PROPERTY_BONUS.get(prop, 0)

    order = np.argsort(-scores)
    return scores[order], ids[order]

def _search_one(query: str, k: int, overfetch: int = 6):
    q = MODEL.encode([query], normalize_embeddings=True)
    scores, I = INDEX.search(q, k * overfetch)
    scores, I = scores[0], I[0]
    scores, I = _priority_boost(scores, I)
    return scores[:k], I[:k]

# ----- 핵심: Procedure Step 1~20 + section별 top-k -----
def retrieve_and_cite_structured(query: str, k_section: int = 2) -> List[str]:
    """
    Procedure Step 1~20을 확보하고, 나머지 section은 상위 k_section개만 선택
    """
    scores, ids = _search_one(query, k=100)
    procedure_steps = {}
    section_topk = {}
    rag_docs = []

    seen_texts = set()

    for s, idx in zip(scores, ids):
        m = META[int(idx)]
        cid = m.get("chunk_id")
        sec = (m.get("section") or "").upper()
        subsec = m.get("subsection")
        txt = (m.get("content") or CHUNK_TEXT.get(cid, "") or "").replace("\n", " ")

        if txt in seen_texts:
            continue
        seen_texts.add(txt)

        # Procedure Step 1~20 안정적 추출
        step = None
        if sec.startswith("PROCEDURE") and subsec:
            match = re.search(r'\d+', subsec)
            if match:
                step = int(match.group(0))
        if step and 1 <= step <= 20:
            procedure_steps[step] = f"[{cid} | {sec} subsec {subsec} · score={s:.3f}] {txt[:500]}"
            continue

        # 나머지 section top-k
        if sec not in section_topk:
            section_topk[sec] = []
        if len(section_topk[sec]) < k_section:
            section_topk[sec].append(f"[{cid} | {sec} {f'subsec {subsec}' if subsec else ''} · score={s:.3f}] {txt[:500]}")

    # Procedure Step 1~20 정렬 후 추가
    for step in range(1, 21):
        if step in procedure_steps:
            rag_docs.append(procedure_steps[step])

    # 나머지 section top-k 추가
    for sec_chunks in section_topk.values():
        rag_docs.extend(sec_chunks)

    return rag_docs

# --- 예시 ---
if __name__ == "__main__":
    query = "How to handle low solubility compounds and DMSO co-solvent?"
    results = retrieve_and_cite_structured(query)
    for r in results:
        print(r)

'''