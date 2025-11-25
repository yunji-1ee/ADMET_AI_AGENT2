# app.py
# 허깅페이스 스페이스에서 실행되는 엔트리 포인트
# src/main_agent_ui.py 안에서 streamlit 코드를 그대로 사용

import os
import sys

# 현재 위치 기준으로 src 폴더를 파이썬 모듈 검색 경로에 추가
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# main_agent_ui를 import 하면, 그 안의 streamlit 코드가 실행됨
import main_agent_ui  # noqa: F401  # (사용은 안 하지만 import만으로 충분)
