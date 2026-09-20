import pandas as pd
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"
APP_ICON = "🧠"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

# 브라우저 탭 제목과 아이콘 (다른 st 명령보다 가장 먼저 호출해야 함)
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

# 숫자 카드의 글자를 크게 보이게 하는 스타일
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 3rem; font-weight: 700; }
    [data-testid="stMetricLabel"] p { font-size: 1.1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def describe_values(series: pd.Series) -> str:
    """열에 어떤 값이 들어 있는지 짧게 설명한다."""
    valid = series.dropna()
    n_unique = valid.nunique()
    if n_unique <= 10:
        return ", ".join(str(v) for v in sorted(valid.unique()))
    if pd.api.types.is_numeric_dtype(series):
        return f"숫자 ({valid.min():g} ~ {valid.max():g})"
    return f"문자 (고유값 {n_unique}개)"


def build_column_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "열 이름": df.columns,
            "우리말 뜻": "",  # 교재를 보고 직접 채워 넣는 칸
            "값의 종류": [describe_values(df[c]) for c in df.columns],
            "빈 값 개수": [int(df[c].isna().sum()) for c in df.columns],
        }
    )


# ---------------------------------------------------------------- 화면 맨 위
st.title(f"{APP_ICON} {APP_TITLE}")
st.caption("뇌졸중 데이터가 무엇인지 먼저 살펴보는 소개 화면입니다.")

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다. 인터넷 연결과 주소를 확인해 주세요.\n\n{e}")
    st.stop()

# ---------------------------------------------------------------- 큰 숫자 카드 4개
total = len(df)
n_cols = df.shape[1]
n_stroke = int((df["stroke"] == 1).sum())
ratio = n_stroke / total * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("전체 사람 수", f"{total:,}명", border=True)
c2.metric("열 개수", f"{n_cols}개", border=True)
c3.metric("stroke가 1인 사람 수", f"{n_stroke:,}명", border=True)
c4.metric("stroke가 1인 비율", f"{ratio:.2f}%", border=True)

# ---------------------------------------------------------------- 열 설명 표
st.subheader("열 설명 표")
st.caption("‘우리말 뜻’ 칸을 더블클릭해서 교재를 보고 직접 채워 넣어 보세요.")

if "column_table" not in st.session_state:
    st.session_state["column_table"] = build_column_table(df)

edited = st.data_editor(
    st.session_state["column_table"],
    key="column_table_editor",
    hide_index=True,
    use_container_width=True,
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름"),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻", help="교재를 보고 직접 적어 주세요."
        ),
        "값의 종류": st.column_config.TextColumn("값의 종류"),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", format="%d"),
    },
)
st.session_state["column_table"] = edited

# ---------------------------------------------------------------- 처음 다섯 줄
st.subheader("데이터 처음 다섯 줄")
st.dataframe(df.head(5), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- 데이터 출처
st.subheader("데이터 출처")
st.text_area(
    "출처",
    key="data_source",
    placeholder="교재에 나온 데이터 출처 글을 여기에 적어 주세요.",
    height=100,
    label_visibility="collapsed",
)
