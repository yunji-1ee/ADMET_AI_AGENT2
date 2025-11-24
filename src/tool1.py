# tool1.py
import re
import math
import requests
from bs4 import BeautifulSoup
from rdkit import Chem
from rdkit.Chem import Descriptors
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

import os
import json
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수에서 키 불러오기
groq_api_key = os.getenv("GROQ_API_KEY")

# 환경변수를 코드에 주입
os.environ["GROQ_API_KEY"] = groq_api_key


# -------------------------
# 1️⃣ GROQ API Key
# -------------------------

# -------------------------
# 2️⃣ PubChem 기반 pKa 추출
# -------------------------
def fetch_pka_from_pubchem(smiles: str):
    try:
        url_search = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{requests.utils.requote_uri(smiles)}/cids/JSON"
        r = requests.get(url_search, timeout=10)
        r.raise_for_status()
        data = r.json()
        cids = data.get("IdentifierList", {}).get("CID", [])
        if not cids:
            return {"source": "none", "pKa_values": []}

        cid = cids[0]
        html_url = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"
        html = requests.get(html_url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=" ")
        pka_values = re.findall(r"pKa\s*[:=]?\s*(-?\d+\.\d+|\d+)", text)

        if pka_values:
            return {"source": "PubChem", "pKa_values": list(map(float, pka_values))}

        heuristics = []
        if "COOH" in smiles or "C(=O)O" in smiles:
            heuristics.append(4.5)
        if "NH2" in smiles:
            heuristics.append(9.5)
        if "OH" in smiles and "CO" in smiles:
            heuristics.append(10.0)
        if not heuristics:
            heuristics.append(7.0)

        return {"source": "Heuristic", "pKa_values": heuristics}
    except Exception:
        return {"source": "error", "pKa_values": []}

# -------------------------
# 3️⃣ LogD 계산
# -------------------------
def calc_logd(logp, pka_values, compound_type="acid"):
    if not pka_values:
        return {"LogD_pH7.4": None, "LogD_pH6.5": None}

    pka = sorted(pka_values, key=lambda x: abs(x - 7.4))[0]

    def logd_at_pH(pH):
        if compound_type == "acid":
            return logp - math.log10(1 + 10 ** (pH - pka))
        elif compound_type == "base":
            return logp - math.log10(1 + 10 ** (pka - pH))
        else:
            return logp

    return {
        "LogD_pH7.4": round(logd_at_pH(7.4), 2),
        "LogD_pH6.5": round(logd_at_pH(6.5), 2)
    }

# -------------------------
# 4️⃣ 연구 목적 + SMILES 추출 (LLM)
# -------------------------
def extract_research_info(text: str):
    smiles_match = re.findall(r"[A-Za-z0-9@+\-\[\]\(\)=#$]{2,}", text)
    smiles_candidates = [s for s in smiles_match if Chem.MolFromSmiles(s)]
    smiles = smiles_candidates[0] if smiles_candidates else None

    llm = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)
    prompt = ChatPromptTemplate.from_template("""
You are an expert in ADME and experimental pharmacokinetics.

Given the following user description, infer the **scientific research objective**
as a clear, professional sentence in English. The objective should reflect
what the user is trying to study or predict (e.g., absorption, metabolism,
permeability, transport, bioavailability, etc.).

Examples:
- "Prediction of oral absorption using Caco-2 permeability assay"
- "Evaluation of intestinal permeability for a new drug candidate"
- "Investigation of transport mechanism across epithelial monolayer"

User input:
{text}

Output only one concise scientific sentence.
""")
    purpose = (prompt | llm).invoke({"text": text}).content.strip()
    return {"purpose": purpose, "smiles": smiles}

# -------------------------
# 5️⃣ 분자 물성 계산
# -------------------------
def calculate_properties(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {"error": "Invalid SMILES string"}

    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rb = Descriptors.NumRotatableBonds(mol)
    ap = sum(atom.GetIsAromatic() for atom in mol.GetAtoms()) / mol.GetNumAtoms()
    logs = 0.16 - (0.63 * logp) - (0.0062 * mw) + (0.066 * rb) - (0.74 * ap)
    tpsa = Descriptors.TPSA(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)

    pka_info = fetch_pka_from_pubchem(smiles)
    logd_info = calc_logd(logp, pka_info["pKa_values"])

    tox_score = 0
    if logp > 5: tox_score += 2
    if logs < -6: tox_score += 2
    if tpsa < 20: tox_score += 1
    if h_donors > 5 or h_acceptors > 10: tox_score += 1
    tox_flag = "High" if tox_score >= 4 else "Medium" if tox_score >= 2 else "Low"

    return {
        "SMILES": smiles,
        "LogP": round(logp, 2),
        "LogS": round(logs, 2),
        "MW": round(mw, 2),
        "TPSA": round(tpsa, 2),
        "H_Donors": h_donors,
        "H_Acceptors": h_acceptors,
        "pKa_source": pka_info["source"],
        "pKa_values": pka_info["pKa_values"],
        **logd_info,
        "toxicity_score": tox_score,
        "toxicity_flag": tox_flag
    }
