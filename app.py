import random
import numpy as np
import math
from numpy.linalg import norm
from gensim.models import KeyedVectors
import streamlit as st
import pandas as pd
import altair as alt
from supabase import create_client, Client

@st.cache_resource
def get_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = get_supabase()

EMBED_PATH = "ko_trimmed.vec"   # 트림된 임베딩 파일
WORDS_PATH = "words.txt"        # 단어 + 빈도 파일

# ===== 임베딩, 단어 로딩 =====
@st.cache_resource
def load_embeddings():
    kv = KeyedVectors.load_word2vec_format(
        EMBED_PATH,
        binary=False,
        encoding="utf-8",
        unicode_errors="ignore",
    )
    return kv

@st.cache_resource
def load_allowed_words():
    words = []
    with open(WORDS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            w = line.split()[0]
            words.append(w)
    return words

def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2) / (norm(v1) * norm(v2) + 1e-9))

def sim_words(kv: KeyedVectors, w1: str, w2: str) -> float:
    return cosine_sim(kv[w1], kv[w2])

def make_bar(pct: float, width: int = 20) -> str:
    """퍼센트(0~100)를 길이 width짜리 막대 그래프로 바꿔주는 함수"""
    pct = max(0.0, min(100.0, pct))  # 0~100으로 클램프
    filled = int(pct / 100 * width)
    empty = width - filled
    return "■" * filled + "□" * empty

def pick_start_goal(kv, candidates, min_sim=0.12, max_sim=0.13, max_trials=5000):
    """유사도 [min_sim, max_sim] 범위인 (start, goal) 뽑기"""
    pool = [w for w in candidates if w in kv.key_to_index]
    if len(pool) < 2:
        raise ValueError("임베딩에 있는 후보 단어가 너무 적음.")

    for _ in range(max_trials):
        start = random.choice(pool)
        goal = random.choice(pool)
        if start == goal:
            continue
        s = sim_words(kv, start, goal)
        if min_sim <= s <= max_sim:
            return start, goal, s

    raise RuntimeError("조건에 맞는 (start, goal)을 찾지 못했습니다. 다시 시작해 주세요.")

def log_game_result(steps: int, success: bool, start_word: str | None = None, goal_word: str | None = None):
    """게임 결과를 Supabase에 기록하는 함수"""
    supabase.table("wordjump-play").insert({
        "user_id": "anonymous",
        "steps": steps,
        "success": success,
        "start_word": start_word,
        "goal_word": goal_word,
    }).execute()
    
sim_threshold = 0.3  # 일반 점프 가능 유사도 기준
goal_threshold = 0.4 # 도착 단어 점프 가능 유사도 기준

# ===== 메인 앱 =====
def main():
    st.set_page_config(page_title="뜻말잇기")

    # 전체 가운데 정렬 CSS
    st.markdown(
        """
        <style>
        div.block-container {
            text-align: center !important;
            max-width: 900px;
            margin-left: auto;
            margin-right: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        /* center all st.button */
        div.stButton > button {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: auto !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    kv = load_embeddings()
    allowed = load_allowed_words()

    # 세션 초기화
    if "game_started" not in st.session_state:
        st.session_state.game_started = False
        st.session_state.start = None
        st.session_state.goal = None
        st.session_state.sg_sim = None
        st.session_state.current = None
        st.session_state.path = []
        st.session_state.messages = []
        st.session_state.user_input = ""
        st.session_state.clear_input = False
        st.session_state.last_warning = ""
        st.session_state.last_success = ""
        
    if "last_warning" not in st.session_state: # 점프 실패 메시지
        st.session_state.last_warning = ""
    if "last_success" not in st.session_state: # 점프 성공 메시지
        st.session_state.last_success = ""

    # 아직 한 번도 게임을 세팅한 적 없으면 자동으로 한 번만 세팅
    if st.session_state.start is None:
        start, goal, sg_sim = pick_start_goal(
            kv,
            allowed,
            min_sim=0.12,
            max_sim=0.13,
        )
        st.session_state.start = start
        st.session_state.goal = goal
        st.session_state.sg_sim = sg_sim
        st.session_state.current = start
        st.session_state.path = [start]
        st.session_state.messages = []
        st.session_state.game_started = True

    # 직전 렌더에서 "입력 비우기" 플래그가 켜져 있었으면 먼저 비우기
    if st.session_state.clear_input:
        st.session_state.user_input = ""
        st.session_state.clear_input = False

    # 상단 텍스트
    st.title("뜻말잇기")
    st.markdown("")
    st.markdown("비슷한 단어들로 점프하며 **출발 단어에서 도착 단어까지** 가 보세요.")
    st.markdown("**2음절 이상, 5음절 이하 명사만** 입력 가능합니다.")
    st.markdown("점프하려면 유사도가 **30% 이상**이어야 하며, 마지막 도착 단어로 점프하려면 **40% 이상**이어야 합니다.")

    # '새 게임' 버튼 가운데 정렬
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        if st.button("새 게임", use_container_width=True):
            start, goal, sg_sim = pick_start_goal(
                kv,
                allowed,
                min_sim=0.12,
                max_sim=0.13,
            )
            st.session_state.start = start
            st.session_state.goal = goal
            st.session_state.sg_sim = sg_sim
            st.session_state.current = start
            st.session_state.path = [start]
            st.session_state.messages = []
            st.session_state.user_input = ""
            st.session_state.clear_input = False
            st.rerun()

    st.write("---")

    # 변수 선언
    start = st.session_state.start
    goal = st.session_state.goal
    current = st.session_state.current
    sg_sim = st.session_state.sg_sim

    # 정답 단어 도착 시 팝업
    if st.session_state.path and st.session_state.path[-1] == goal:
        steps = len(st.session_state.path) - 1  # 점프 횟수
        path_str = " → ".join(st.session_state.path) # 점프 경로
        st.markdown(
            f"""
            <div style="
                display: flex;
                justify-content: center;
                margin-bottom: 20px;
            ">
                <div style="
                    background-color: #d4edda;
                    color: #155724;
                    padding: 15px 25px;
                    border-radius: 8px;
                    border: 1px solid #c3e6cb;
                    width: 50%;          /* 가로폭 조절 */
                    max-width: 500px;    /* 최대 가로폭 */
                    text-align: center;
                    font-size: 18px;
                ">
                    🎉 <b>{goal}</b>에 도착했습니다!<br><br>
                    <b>점프 횟수</b>: {steps}번<br><br>
                    <b>점프 경로</b>: {path_str}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # 출발 단어와 도착 단어 박스로 만들기
    st.markdown(
        f"""
<div style="background-color:#f5f5f5; padding:15px 20px; border-radius:10px; border:1px solid #ddd; display:inline-block; font-size:15px; margin-bottom:15px;">
  <b>[출발 단어]</b>
  <a href="https://www.google.com/search?q={start}" target="_blank" style="color:inherit; text-decoration:none;">{start}</a>
  &nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;
  <b>[도착 단어]</b>
  <a href="https://www.google.com/search?q={goal}" target="_blank" style="color:inherit; text-decoration:none;">{goal}</a>
</div>
        """,
        unsafe_allow_html=True
    )

    # 지나온 단어들의 유사도 그래프
    if st.session_state.path and goal is not None:
        sims = [sim_words(kv, w, goal) * 100 for w in st.session_state.path]

        # 데이터프레임 만들기
        df_sims = pd.DataFrame({
            "step": list(range(len(sims))),   # 0,1,2,... 순서
            "word": st.session_state.path,    # 단어
            "similarity": sims,               # 유사도(%)
        })
        
        # x축 라벨로 쓸 단어 배열을 JS 표현식으로 만들기
        labels_js = "[" + ", ".join(f"'{w}'" for w in st.session_state.path) + "]"

        st.markdown(f"**[{goal}와(과)의 유사도]**")

        # 그래프 구성
        chart = (
            alt.Chart(df_sims)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "step:Q",
                    title=None,
                    axis=alt.Axis(
                        grid=True,
                        labelAngle=-30,
                        labelExpr=f"{labels_js}[datum.value]"  # step을 인덱스로 단어 꺼내기
                    ),
                ),
                y=alt.Y(
                    "similarity:Q",
                    title="유사도(%)",
                    scale=alt.Scale(domain=[0, 100]),
                ),
                tooltip=["word", "similarity"],
            )
            .properties(height=200)
        )

        st.altair_chart(chart, use_container_width=True)

    # 입력란
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.session_state.current != st.session_state.goal:
            with st.form("move_form"):
                
                # 점프 성공 메시지
                if st.session_state.last_success:
                    text = st.session_state.last_success.replace("\n", "<br>")
                    st.markdown(
                        f"<span style='color:#1f77b4;'>{text}</span>",
                        unsafe_allow_html=True,
                    )
                    
                # 레이블을 따로 빼서 가운데 정렬
                st.markdown(f"**[현재 단어: {current}]**")

                # 실제 입력칸은 라벨 없이
                user = st.text_input(
                    "",
                    key="user_input",
                )
                
                if st.session_state.last_warning:
                    text = st.session_state.last_warning.replace("\n", "<br>")
                    st.markdown(
                        f"<span style='color:#E03C31;'>{text}</span>",
                        unsafe_allow_html=True,
                    )

                # 점프 버튼 (가운데 정렬, 가로폭 좁게)
                b1, b2, b3 = st.columns([2, 1, 2])
                with b2:
                    submit = st.form_submit_button("점프")

            # 입력란에 자동 포커스
            st.markdown(
                """
                <script>
                const inputs = window.parent.document.querySelectorAll('input[type="text"]');
                if (inputs.length > 0) {
                    const input = inputs[inputs.length - 1];
                    input.focus();
                    input.select();
                }
                </script>
                """,
                unsafe_allow_html=True,
            )

    # 점프 시도 처리
    if 'submit' in locals() and submit:
        user = user.strip()
        current = st.session_state.current
        # 경고/성공 메시지 비워두기
        st.session_state.last_warning = ""
        st.session_state.last_success = ""
        
        # 유저 입력 처리
        
        # 불량입력 시
        if user == "":
            st.session_state.last_warning = "단어를 입력하세요."
        elif len(user) < 2 or len(user) > 5:
            st.session_state.last_warning = "2음절 이상, 5음절 이하만 입력 가능합니다."
        elif user == current: 
            st.session_state.last_warning = "현재 단어와 다른 단어를 입력하세요."
        elif (user not in allowed) or (user not in kv.key_to_index):
            st.session_state.last_warning = "너무 일반적이거나 없는 단어입니다. 다른 단어로 입력해 보세요."
        
        # 정상입력 시
        else:
            current = st.session_state.current
            sim_cur = sim_words(kv, current, user)
            
            # 이번 점프에서 요구되는 유사도 기준
            # 정답 단어로 점프할 경우 goal_threshold 적용
            required = goal_threshold if user == goal else sim_threshold
            
            # 점프 실패
            if sim_cur < required:
                # 0.0 ~ 1.0 범위의 sim_cur를 퍼센트로 바꾼 뒤 소수 첫째 자리에서 버림
                sim_pct = math.floor(sim_cur * 1000) / 10  # 예: 0.349 → 34.9
                st.session_state.last_warning = (
                    f"{user}은(는) 관련이 적은 단어입니다.\n유사도: {sim_pct:.1f}%"
                )
            
            # 점프 성공
            else:
                # current 갱신
                previous = current
                st.session_state.current = user
                st.session_state.path.append(user)

                sim_to_goal = sim_words(kv, user, goal)
                
                # 유사도를 퍼센트로 변환
                cur_pct = sim_cur * 100
                goal_pct = sim_to_goal * 100

                # 간단한 텍스트 막대 그래프
                bar_cur = make_bar(cur_pct, width=20)
                bar_goal = make_bar(goal_pct, width=20)

                # 소수 첫째 자리에서 버림한 퍼센트 값
                cur_pct_trunc = math.floor(cur_pct * 10) / 10
                goal_pct_trunc = math.floor(goal_pct * 10) / 10
                
                msg = (
                    f"**[{user}의 유사도]**\n\n"
                    f"**{previous}**: {cur_pct_trunc:.1f}% {bar_cur}\n\n"
                    f"**{goal}**: {goal_pct_trunc:.1f}% {bar_goal}"
                )
                st.session_state.messages.append(msg)
                
                # 도착 단어 도달 메시지
                if user == goal:
                    steps = len(st.session_state.path) - 1
                    log_game_result(
                        steps=steps,
                        success=True,
                        start_word=st.session_state.start,
                        goal_word=st.session_state.goal
                    )
                    st.session_state.game_started = False
    
                # 점프 성공 메시지
                else:
                    sim_cur_pct_trunc = math.floor(sim_cur * 1000) / 10
                    st.session_state.last_success = (
                        f"**{previous}** → **{user}** 점프! (유사도: {sim_cur_pct_trunc:.1f}%)"
                    )

        # 다음 렌더에서 입력칸 비우기
        st.session_state.clear_input = True
        st.rerun()

    # 하단 설명과 저작권 표시
    st.markdown(
        """
        <div style='margin-top:50px; color:#777; font-size:14px; text-align:center;'>
            유사도는 비슷한 맥락에서 함께 자주 쓰이는 단어들끼리 높게 책정됩니다.<br>
            게임의 재미를 위해, '사람'처럼 너무 일반적인 단어나 '해'와 같은 1음절 단어는 입력할 수 없습니다.<br>
            이러한 단어들은 지나치게 많은 단어와 유사도가 높게 책정되기 때문입니다.<br><br>
            생소한 단어가 출발 단어나 도착 단어로 주어질 수 있습니다.<br>모르는 단어는 눌러 보세요. '새 게임'도 적극 권장합니다.<br><br>
            ───────────────────────────────────────<br>
            <b>뜻말잇기</b> · Korean Word Jump Game<br><br>
            임베딩: FastText 한국어 벡터 (cc. BY-SA)<br>
            단어 목록: 국립국어원 표준국어대사전 표제어<br>(2003 국립국어원 빈도 자료를 참고해 재구성)<br><br>
            © 2025 fightingduck · <a href="https://github.com/jeontuori/wordjump-game" target="_blank" style="color:#777; text-decoration:none;">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True
    )
if __name__ == "__main__":
    main()