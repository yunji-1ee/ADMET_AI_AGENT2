# -*- coding: utf-8 -*-
import os, re, json
import streamlit as st
from dotenv import load_dotenv

from tool1 import extract_research_info, calculate_properties

try:
    from tool2 import retrieve_and_cite_structured
except Exception:
    retrieve_and_cite_structured = None

try:
    from tool2 import retrieve_adme_protocols
except Exception:
    retrieve_adme_protocols = None

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# =========================
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = groq_api_key or ""
# API 설정 여부 플래그
API_READY = bool(groq_api_key)

st.set_page_config(
    page_title=" AI-ADMET-AGENT ",
    page_icon="🧬",
    layout="wide"     # 화면 넓게 사용
)

# =========================
# 전역 스타일 (CSS)
# =========================
st.markdown(
    """
    <style>
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background-color: #f6f7fb;
    }
    .main > div {
        padding-top: 0rem;
    }
    .aep-card {
        background: #ffffff;
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        border: 1px solid rgba(148, 163, 184, 0.25);
    }
    .aep-badge {
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        background: #eef2ff;
        color: #4338ca;
        display: inline-block;
        margin-bottom: 0.4rem;
    }
    /* SOP 범위별 색상 배지 */
    .aep-badge-ok {
        background: #dcfce7;
        color: #166534;
    }
    .aep-badge-warn {
        background: #fef9c3;
        color: #854d0e;
    }
    .aep-badge-out {
        background: #fee2e2;
        color: #b91c1c;
    }

    .aep-section-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .aep-section-sub {
        font-size: 0.85rem;
        color: #6b7280;
        margin-bottom: 0.8rem;
    }
    .aep-label {
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)




# =========================
def safe_extract_rag_query(raw: str) -> dict:
    if not raw or not str(raw).strip():
        return {"rag_query": ""}
    s = str(raw).strip()
    s = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", s, flags=re.I | re.M)
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        s = s[i:j+1]
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            if isinstance(obj.get("rag_query"), str) and obj["rag_query"].strip():
                return {"rag_query": obj["rag_query"].strip()}
            for v in obj.values():
                if isinstance(v, str) and v.strip():
                    return {"rag_query": v.strip()}
        if isinstance(obj, str) and obj.strip():
            return {"rag_query": obj.strip()}
    except Exception:
        pass
    first_line = s.splitlines()[0].strip()
    return {"rag_query": first_line}

# =========================
def generate_rag_keywords(purpose: str, props: dict) -> str:
    llm = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)
    prompt_template = ChatPromptTemplate.from_template("""
You are an expert in pharmacokinetics and ADME optimization.

Your task is to generate a **compact RAG query keyword string** suitable for retrieving relevant ADME experimental protocols.
You will base your reasoning on both the **research objective** and the **calculated molecular properties**, explicitly comparing each property to the **SOP standard ranges**.

---

 **SOP Standard Ranges**
- LogP: 0 ~ 3  
- MW: < 500  
- LogS: -3 ~ -1  
- TPSA: < 100  
- Toxicity: Low (toxicity_score 0~1)  
- pKa: 6.5 ~ 7.4  

---

 **Calculated Properties**
{props_json}

 **Research Objective**
{text_obj}

---

Instructions for constructing the query:

1 Include the **research objective exactly as given by the user** (do not rephrase).
2 For each property, assess its deviation from SOP ranges:
   - Label as "low", "high", "standard", or "out-of-range"
   - Highlight unusual cases explicitly (e.g., "high MW", "low LogS")
3 For the query, **request actionable information**, including:
   - Step-by-step experimental procedures (including any special conditions)
   - Subsection structure (e.g., Cultivation vs Experimental procedure)
   - Any adjustments required due to deviations from SOP
4 Construct the query as a **natural request** for the RAG system, e.g.:
   - “Please provide experimental procedures for Caco-2 assay considering these property deviations…”
5 Output the query as **JSON object only**, like:
{{"rag_query": "Detailed request for ADME experimental protocols with steps and deviations highlighted."}}
6 **Do not output code fences, explanations, or multiple JSON objects.**
""")
    resp = (prompt_template | llm).invoke({
        "text_obj": purpose,
        "props_json": json.dumps(props, ensure_ascii=False)
    })
    return resp.content.strip()

# =========================
def generate_experimental_guideline(rag_query: str, rag_docs: list, props_json: dict, research_objective: str) -> str:
    llm = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.3)
    prompt = ChatPromptTemplate.from_template("""
You are a **pharmacokinetics and ADME laboratory expert**.

You have access to:
- The **user's original research objective**
- The **calculated molecular properties (Tool 1 JSON)**
- The **retrieved experimental text snippets (Tool 2 RAG results)**

Your task:
After receiving all text snippets from Tool 2, **stop calling tools** and synthesize *all* the information you have 
into a **final structured, detailed, and practical experimental design report**.

---

### Context

**Research Objective (verbatim from user):**
{research_objective}

**Molecular Properties (from Tool 1):**
{props_json}

**Retrieved Experimental Snippets (from Tool 2):**
{rag_docs}

---

### Final Output Structure (must strictly follow this format)

**1. Research Objective:**  
- Clearly restate the user's research goal. Do not paraphrase, only clarify if needed.

**2. Molecular Characteristics:**  
- Summarize the molecule’s ADME-relevant properties from Tool 1 JSON.  
- Interpret briefly whether each property (LogP, MW, LogS, TPSA, toxicity, pKa) is low/high/standard/out-of-range.

**3. Experimental Implications:**  
- Explain *why* standard SOP procedures may or may not be suitable for this molecule.  
- Reference both Tool 1 JSON and RAG snippets to justify adaptations.  
- Include specific insights (e.g., “low solubility suggests longer dissolution time”, “high TPSA implies lower permeability”).  
+ ⚠ **Explicitly identify all deviations from SOP based on molecular properties.**  
+ ⚠ **Prioritize Modified Steps from RAG snippets for unusual cases.**  
+ ⚠ **For each deviation, provide reasoning and concrete adjustment suggestions.**

**4. Experimental Procedure (structured, practical):**  
Provide a **concise, step-wise experimental workflow** optimized for the molecule.  
Organize under these subheaders:
   - **Reagents:** (List all chemical reagents and buffer systems typically required)  
+        - **Ensure all Caco-2 cell culture reagents are included (e.g., DMEM, FBS, Trypsin/EDTA, Nonessential Amino Acids).**
+        - **Include specific reagents for property adjustments (e.g., DMSO, BSA, Mannitol).**
   - **Equipment:** (List essential instruments, e.g., Caco-2 plates, LC-MS, incubator, etc.)  
+        - **Crucially include Trans-Epithelial Electrical Resistance (TER) measurement device (e.g., Endohm chamber with voltmeter).**
+        - **Specify the type of Caco-2 plates (e.g., 12-well Transwell inserts).**
   - **Setup:** (Describe pre-experiment setup — e.g., cell seeding density, pre-incubation time, solvent prep)  
+        - **Specify standard Caco-2 cell seeding density (e.g., 2.6 x 10^5 cells/cm^2) and justify any deviations based on molecular properties.**
+        - **State the pre-incubation/differentiation time clearly (e.g., 21-29 days).**
+        - **Describe the preparation of donor and receiver solutions, including any co-solvents (DMSO) or additives (BSA) with their final concentrations and placement (e.g., "DMSO max 1% in both compartments", "4% BSA in receiver compartment").**
+        - ⚠ **Ensure all Modified Steps identified in RAG snippets are incorporated here and clearly referenced in the Experimental Steps.**
   - **Experimental Steps (20 steps, detailed and precise):**  
        Provide 20 **clear, numbered steps** outlining the optimized experimental procedure for the molecule.  
+        **These steps must fully cover the entire Caco-2 assay workflow, from initial cell culture preparation to final data interpretation.**
+        **Crucial steps to include are:**
+        - **Caco-2 cell thawing, culturing, trypsinization, counting, and seeding onto permeable supports.**
+        - **Regular media changes for 21-29 days to ensure differentiation.**
+        - **Pre-experiment washing of monolayers with HBSS.**
+        - **Trans-Epithelial Electrical Resistance (TER) measurement (before and after transport experiment).**
+        - **Cell monolayer integrity test (e.g., [14C]Mannitol permeability assay) as a quality control and for toxicity assessment.**
+        - **Preparation of donor and receiver solutions with specific pH (e.g., pH 7.4, or pH 6.5 for apical side if needed).**
+        - **Performing transport experiments in both apical-to-basolateral (absorptive) AND basolateral-to-apical (secretory) directions for efflux ratio calculation.**
+        - **Incubation on an orbital shaker at specified RPM.**
+        - **Sampling from the receiver compartment at multiple time points, specifying sample volume and replacement with fresh buffer.**
+        - **Final sample collection from the donor compartment for mass balance calculation.**
+        - **Analytical method (LC-MS) for quantification.**
+        - **Calculation of apparent permeability coefficient (Papp).**
+        - **Calculation of efflux ratio/uptake ratio for transport mechanism evaluation.**
+        - **Data interpretation.**
        - If the molecule deviates from SOP, describe how each step is modified accordingly.  
        - Each step should be a precise action sentence (e.g., “Incubate Caco-2 cells at 37°C for 1h in HBSS buffer”).  
        - Highlight any modified steps with clear justification (e.g., “⚠ Adjusted due to high MW or low solubility”).  
        - Steps should be complete and practically implementable; avoid overly short summaries.

**5. Modified Steps(special case):** 
- Ensure consistency between general steps and specific adjustments outlined in this 'Modified Steps' section.
- Explicitly include any special steps or deviations from SOP, such as:
    -> Mannitol integrity test for monolayer quality control
    -> DMSO/BSA concentration adjustments based on solubility or protein binding
    -> Handling of low solubility or poorly soluble compounds
    -> High molecular weight, high lipophilicity, or toxicity-driven adaptations
- All Modified Steps must be fully integrated into the Experimental Steps section.
    -> Use clear annotations, warning symbols (⚠), or inline comments to indicate the rationale for each modification.
    -> Link each Modified Step directly to the corresponding property deviation or special condition from Tool 1 JSON or RAG snippets.
    -> Ensure no Modified Step is omitted; if a step alters SOP, explicitly describe how and why it is modified.
    -> Maintain chronological order in Experimental Steps while reflecting all necessary adjustments.
- Provide practical, actionable instructions for each Modified Step that can be directly executed in the lab.
- Emphasize safety, accuracy, and reproducibility, clearly marking steps that require special attention or monitoring.

**6. Formal Request:**
End with a single formal statement asking for a tailored plan:  
> “Please provide a detailed experimental setup tailored to the above molecule and research objective, considering all deviations from SOP.”

---

### Output Rules

- Use **precise scientific tone**, no unnecessary text.
- Reference experimental insights from Tool 2 snippets *whenever relevant*.
- Avoid generic filler; emphasize **reasoning + justification** behind modifications.
- ⚠ **Explicitly ensure all deviations from SOP identified in Tool 1 and RAG snippets are included in Modified Steps and Experimental Steps.**
- Output should be complete in one message, formatted clearly in Markdown.

Now generate the final structured report accordingly.
""")

    rag_block = "\n\n".join(rag_docs) if rag_docs else "(no snippets)"
    resp = (prompt | llm).invoke({
        "research_objective": research_objective,
        "props_json": json.dumps(props_json, ensure_ascii=False, indent=2),
        "rag_docs": rag_block
    })
    return resp.content.strip()



# =========================
# SOP 범위 판정 + 하이라이트 카드 렌더링
# =========================

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def classify_props_for_sop(props: dict) -> dict:
    """
    Tool1에서 계산된 물성값을 SOP 기준과 비교해서
    status(in / warn / out)와 라벨 텍스트를 만들어주는 함수.
    """
    result = {}

    # LogP (0~3)
    lp = _to_float(props.get("LogP"))
    if lp is not None:
        if 0 <= lp <= 3:
            status, label = "in", f"{lp:.2f} (within SOP)"
        elif lp < 0:
            status, label = "out", f"{lp:.2f} (low, < 0)"
        else:
            status, label = "warn", f"{lp:.2f} (high, > 3)"
        result["LogP"] = {
            "value": f"{lp:.2f}",
            "status": status,
            "label": label,
            "sop": "0 – 3",
        }

    # LogS (-3 ~ -1)
    ls = _to_float(props.get("LogS"))
    if ls is not None:
        if -3 <= ls <= -1:
            status, label = "in", f"{ls:.2f} (within SOP)"
        elif ls < -3:
            status, label = "out", f"{ls:.2f} (too low, < -3)"
        else:
            status, label = "warn", f"{ls:.2f} (high, > -1)"
        result["LogS"] = {
            "value": f"{ls:.2f}",
            "status": status,
            "label": label,
            "sop": "-3 – -1",
        }

    # MW (< 500)
    mw = _to_float(props.get("MW"))
    if mw is not None:
        if mw < 500:
            status, label = "in", f"{mw:.2f} (within SOP)"
        else:
            status, label = "warn", f"{mw:.2f} (high, ≥ 500)"
        result["MW"] = {
            "value": f"{mw:.2f}",
            "status": status,
            "label": label,
            "sop": "< 500",
        }

    # TPSA (< 100)
    tpsa = _to_float(props.get("TPSA"))
    if tpsa is not None:
        if tpsa < 100:
            status, label = "in", f"{tpsa:.2f} (within SOP)"
        else:
            status, label = "warn", f"{tpsa:.2f} (high, ≥ 100)"
        result["TPSA"] = {
            "value": f"{tpsa:.2f}",
            "status": status,
            "label": label,
            "sop": "< 100",
        }

    # toxicity_flag (Low / Medium / High)
    tox = props.get("toxicity_flag")
    if tox is not None:
        t = str(tox).strip().lower()
        if t == "low":
            status, label = "in", "Low toxicity"
        elif t in ("medium", "moderate"):
            status, label = "warn", tox
        else:
            status, label = "out", tox
        result["toxicity_flag"] = {
            "value": str(tox),
            "status": status,
            "label": label,
            "sop": "Low (0–1)",
        }

    return result


def render_prop_highlight_cards(sop_info: dict):
    """
    SOP 판정 결과를 요약 카드 형태로 화면 상단에 렌더링.
    """
    if not sop_info:
        return

    st.markdown("##### 핵심 ADMET 지표")

    cols = st.columns(len(sop_info))
    for col, (name, info) in zip(cols, sop_info.items()):
        status = info.get("status", "in")
        if status == "in":
            cls, icon = "aep-badge-ok", "✅"
        elif status == "warn":
            cls, icon = "aep-badge-warn", "⚠️"
        else:
            cls, icon = "aep-badge-out", "⛔"

        with col:
            st.markdown(
                f"""
                <div class="aep-card" style="padding:0.9rem 1rem; margin-bottom:0.6rem;">
                  <div class="aep-label">{name}</div>
                  <div style="font-size:1.1rem; font-weight:700; margin:0.1rem 0 0.25rem 0;">
                    {info.get('value', '-')}
                  </div>
                  <span class="aep-badge {cls}">{icon} {info.get('label','')}</span>
                  <div style="font-size:0.75rem; color:#6b7280; margin-top:0.3rem;">
                    SOP: {info.get('sop','-')}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# =========================
# 리포트 텍스트 섹션 분할 (expander용)
# =========================

def split_guideline_sections(text: str):
    """
    한국어 주석: LLM이 만들어준 guideline 문자열을
    1~6번 섹션 기준으로 잘라 리스트로 돌려준다.
    """
    anchors = [
        ("1. Research Objective", "**1. Research Objective:**"),
        ("2. Molecular Characteristics", "**2. Molecular Characteristics:**"),
        ("3. Experimental Implications", "**3. Experimental Implications:**"),
        ("4. Experimental Procedure", "**4. Experimental Procedure"),
        ("5. Modified Steps", "**5. Modified Steps"),
        ("6. Formal Request", "**6. Formal Request:**"),
    ]

    positions = []
    for title, marker in anchors:
        idx = text.find(marker)
        if idx != -1:
            positions.append((idx, title, marker))

    positions.sort(key=lambda x: x[0])
    if not positions:
        return []

    sections = []
    for i, (idx, title, marker) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        section_text = text[idx:end].strip()
        sections.append((title, section_text))

    return sections


# =========================
# 레이아웃: 사이드바 + 메인
# =========================

with st.sidebar:
    # 왼쪽 사이드바 = 연구 입력 영역 (STEP 1)
    st.markdown("###  🤖 ADMET 실험 어시스턴트 🤖")

    st.caption("연구 목적과 SMILES를 한 번에 입력하면, 오른쪽에서 전체 파이프라인 결과를 확인할 수 있습니다.")

    user_input = st.text_area(
        "연구 목적과 SMILES를 입력하세요:",
        placeholder="예) Caco-2 투과도 평가를 위해 다음 후보물질을 테스트: O=C(O)COCCN1CCN(C(c2ccccc2)c2ccc(Cl)cc2)CC1",
        height=220,  # 사이드바 폭 감안해서 높이만 살짝 늘림
    )

    # 결과 변수 기본값
    props = None
    rag_query_obj = None

    # 전체 파이프라인 실행 버튼
    run_pipeline = st.button(
        "ADMET 실험 절차 생성",
        use_container_width=True,
        disabled=not API_READY,   # Groq API 없으면 버튼 비활성화
    )

if not API_READY:
    st.warning(
        "⚠️ GROQ_API_KEY가 설정되지 않았습니다. .env 파일에서 GROQ_API_KEY를 등록한 뒤 다시 실행해주세요.",
        icon="⚠️",
    )


st.markdown("##  ADMET Experimental Planner")
st.caption("연구 텍스트와 SMILES를 기반으로 Caco-2 중심 ADMET 실험 절차를 제안합니다.")
st.markdown("")

# 한국어 주석: 메인 영역 = STEP2 결과 전용, 탭 3개
st.markdown("#### 물성 계산 & RAG 기반 실험 설계")
st.markdown(
    '<div class="aep-section-sub">Tool1의 물성 계산 결과와, '
    '해당 정보를 반영한 RAG 검색 및 최종 Caco-2 실험 설계를 제공합니다.</div>',
    unsafe_allow_html=True
)

# 탭 설정
tab_summary, tab_report, tab_props_debug, tab_rag_debug = st.tabs(
    ["🔍 요약 보기", "💡 최종 실험 제안", " <물성 디버그>", " <RAG 디버그>" ]
)

# --- 실행 전 안내 placeholder ---
if not run_pipeline:
    with tab_summary:
        st.info("왼쪽 step1에서 연구 목적과 SMILES를 입력한 뒤, 'ADMET 실험 절차 생성' 버튼을 눌러주세요.")
    with tab_props_debug:
        st.info("실행 후 Tool1 기반 물성 계산 상세 정보를 확인할 수 있습니다.")
    with tab_rag_debug:
        st.info("실행 후 RAG 검색 과정(LLM 쿼리 및 tool2 결과)을 확인할 수 있습니다.")
    with tab_report:
        st.info("실행 후 이 탭에서 최종 Caco-2 실험 설계 리포트를 볼 수 있습니다.")
        

if run_pipeline:
    # --- 입력 공백 체크 ---
    if not user_input.strip():
        st.error("입력이 비어 있습니다. 연구 목적과 SMILES를 함께 입력해주세요.")
        st.stop()

    # --- Groq API 없는 경우 추가 방어 (버튼은 이미 disabled지만, 혹시 몰라 한 번 더 체크) ---
    if not API_READY:
        st.error("GROQ_API_KEY가 설정되지 않아 파이프라인을 실행할 수 없습니다.")
        st.stop()

    # --- 진행 상태 표시용 컴포넌트 ---
    status_box = st.empty()      # 한국어 주석: 현재 단계 텍스트용 영역
    progress_bar = st.progress(0)  # 한국어 주석: 전체 진행률 표시

    # ===== 1) 연구목표/SMILES 추출 (Tool1) =====
    with st.spinner("step 1/4: 연구 텍스트에서 연구 목적과 SMILES를 추출하는 중입니다…"):
        info = extract_research_info(user_input)
        smiles = info.get("smiles")
        purpose = info.get("purpose")

    if not smiles:
        st.error(
            "유효한 SMILES를 찾을 수 없습니다. 입력 텍스트 안에 SMILES 문자열이 포함되어 있는지 확인해주세요.\n\n"
            "예시: `CCO`, `O=C(O)COCCN1CCN(C(c2ccccc2)c2ccc(Cl)cc2)CC1` 처럼 SMILES가 한 번은 들어가야 합니다."
        )
        st.stop()

    progress_bar.progress(25)
    status_box.info("✅ step 1/4 완료: 연구 목적/SMILES 파싱 완료")

    # ===== 2) 물성 계산 (Tool1) =====
    with st.spinner("step 2/4: RDKit 기반 물성(LogP, LogS, MW, TPSA 등)을 계산하는 중입니다…"):
        props = calculate_properties(smiles)

    progress_bar.progress(50)
    status_box.info("✅ step 2/4 완료: 물성 계산 완료")

    # ===== 3) RAG 쿼리 생성 (LLM) + 안전 파싱 =====
    with st.spinner("step 3/4: 물성 정보를 반영한 RAG 쿼리를 LLM으로 생성하는 중입니다…"):
        rag_keywords_raw = generate_rag_keywords(purpose, props)
        rag_query_obj = safe_extract_rag_query(rag_keywords_raw)

    if not rag_query_obj.get("rag_query"):
        st.error("RAG 쿼리를 해석하지 못했습니다. LLM 응답 형식을 확인해주세요.")
        st.stop()

    query = rag_query_obj["rag_query"]
    progress_bar.progress(70)
    status_box.info("✅ step 3/4 완료: RAG 쿼리 생성 및 파싱 완료")

    # ===== 4) RAG 검색 (tool2) =====
    with st.spinner("step 4/4: RAG 인덱스에서 관련 Caco-2 프로토콜을 검색하고, 최종 실험 설계를 생성하는 중입니다…"):
        rag_docs = []
        if retrieve_and_cite_structured is not None:
            try:
                rag_result = retrieve_and_cite_structured(query, k_section=4)
                # Procedure Steps
                rag_docs.extend(rag_result.get("procedure_steps", []))
                # Section Top-k
                rag_docs.extend(rag_result.get("section_topk", []))
                # Modified Steps
                rag_docs.extend(rag_result.get("modified_steps", []))
            except Exception as e:
                st.warning(f"retrieve_and_cite_structured 실패: {e}")
                rag_docs = []

        guideline = generate_experimental_guideline(
            rag_query=query,
            rag_docs=rag_docs,
            props_json=props,
            research_objective=purpose
        )

    progress_bar.progress(100)
    status_box.success(" 전체 파이프라인 실행이 완료되었습니다.")

    # ===== 5) 탭별 UI 출력 =====

     # --- 요약 탭 ---
    with tab_summary:


        summary_props = {
            "SMILES": props.get("SMILES"),
            "LogP": props.get("LogP"),
            "LogS": props.get("LogS"),
            "MW": props.get("MW"),
            "TPSA": props.get("TPSA"),
            "toxicity_flag": props.get("toxicity_flag"),
        }

        # 1) SOP 기준 하이라이트 카드
        sop_info = classify_props_for_sop(summary_props)
        render_prop_highlight_cards(sop_info)

        # 2) 원래 JSON 요약도 아래에 그대로 유지
        with st.expander("Raw JSON 보기 (계산된 물성 전체)", expanded=False):
            st.json(summary_props)

        st.subheader("실험 설계 핵심 요약")
        preview = "\n".join(guideline.splitlines()[:20])
        st.markdown(preview)
        st.caption("자세한 프로토콜은 ‘💡 최종 실험 제안’ 탭에서 확인하세요.")


    # --- 물성 디버그 탭 ---
    with tab_props_debug:
        st.markdown("### SMILES 파싱 결과")
        st.write({"purpose": purpose, "smiles": smiles})
        st.markdown("### 계산된 물성 (전체 JSON)")
        st.json(props)

    # --- RAG 디버그 탭 ---
    with tab_rag_debug:

        st.markdown("### RAG Query (LLM Raw)")
        st.code(rag_keywords_raw if rag_keywords_raw else "<empty>")
        st.markdown("### Parsed RAG Query")
        st.json(rag_query_obj)
        st.markdown("### RAG 검색 결과 (tool2)")
        for i, s in enumerate(rag_docs, 1):
            st.write(f"{i}. {s[:400]}{'…' if len(s) > 400 else ''}")

    # --- 최종 리포트 탭 ---
        # --- 최종 리포트 탭 ---
    with tab_report:

        # 0) 전체 워크플로우 타임라인 느낌의 개괄
        st.markdown("##### Caco-2 Assay Workflow Overview")
        st.markdown(
            """
            1. **준비(Preparation)** – Reagents / Equipment 준비 및 용액 제조  
            2. **세포 배양(Cell Culture & Seeding)** – Caco-2 thawing, 확장 배양, Transwell seeding  
            3. **Pre-test & QC** – TER 측정, Mannitol 등으로 monolayer integrity 확인  
            4. **Transport Assay** – A→B, B→A 방향 투과 실험 수행 및 샘플링  
            5. **분석(Analysis)** – LC-MS 등으로 농도 분석, Papp 및 efflux ratio 계산  
            6. **데이터 해석(Data Interpretation)** – SOP 대비 결과 해석 및 특이점 검토  
            """
        )

        st.markdown("---")

        # 1) guideline을 섹션별로 expander에 담기
        sections = split_guideline_sections(guideline)

        if not sections:
            # 혹시 파싱이 잘 안됐을 때는 전체를 한 번에 보여줌
            st.markdown(guideline)
        else:
            for title, text in sections:
                # 연구목표/물성은 기본 펼침, 나머지는 접기
                expanded = title.startswith("1.") or title.startswith("2.")
                with st.expander(title, expanded=expanded):
                    st.markdown(text)

        st.markdown("---")
        st.subheader("결과 내보내기")

        # Markdown 파일 다운로드
        md_bytes = guideline.encode("utf-8")
        st.download_button(
            label="Markdown (.md)로 저장",
            data=md_bytes,
            file_name="aep_caco2_protocol.md",
            mime="text/markdown",
        )

        