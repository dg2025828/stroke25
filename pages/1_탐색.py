import pandas as pd
import plotly.express as px
import streamlit as st

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

st.set_page_config(page_title="탐색 - 뇌졸중 예측 실습실", page_icon="🔍", layout="wide")


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def stroke_rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """col의 값(0/1)별 뇌졸중 비율(%)을 계산한다."""
    rate = df.groupby(col)["stroke"].agg(["mean", "size"]).reset_index()
    rate["뇌졸중 비율(%)"] = (rate["mean"] * 100).round(2)
    rate["구분"] = rate[col].map({0: "없음", 1: "있음"})
    return rate.rename(columns={"size": "사람 수"})


st.title("🔍 탐색")
st.caption("데이터를 여러 방향에서 살펴보는 페이지입니다.")

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다. 인터넷 연결과 주소를 확인해 주세요.\n\n{e}")
    st.stop()

# ---------------------------------------------------------------- 1. 분포 (히스토그램)
st.header("1. 나이와 평균 혈당의 분포")
left, right = st.columns(2)

fig_age = px.histogram(
    df, x="age", nbins=30, title="나이 분포",
    labels={"age": "나이", "count": "사람 수"},
)
fig_age.update_layout(yaxis_title="사람 수")
left.plotly_chart(fig_age, use_container_width=True)

fig_glu = px.histogram(
    df, x="avg_glucose_level", nbins=40, title="평균 혈당 분포",
    labels={"avg_glucose_level": "평균 혈당", "count": "사람 수"},
)
fig_glu.update_layout(yaxis_title="사람 수")
right.plotly_chart(fig_glu, use_container_width=True)

# ---------------------------------------------------------------- 2. 그룹 비교 (상자그림)
st.header("2. 뇌졸중 여부에 따른 나이와 평균 혈당")

plot_df = df.copy()
plot_df["뇌졸중 여부"] = plot_df["stroke"].map({0: "겪지 않음", 1: "겪음"})
order = {"뇌졸중 여부": ["겪지 않음", "겪음"]}

left, right = st.columns(2)
fig_box_age = px.box(
    plot_df, x="뇌졸중 여부", y="age", color="뇌졸중 여부",
    category_orders=order, title="나이 비교", labels={"age": "나이"},
)
fig_box_age.update_layout(showlegend=False)
left.plotly_chart(fig_box_age, use_container_width=True)

fig_box_glu = px.box(
    plot_df, x="뇌졸중 여부", y="avg_glucose_level", color="뇌졸중 여부",
    category_orders=order, title="평균 혈당 비교",
    labels={"avg_glucose_level": "평균 혈당"},
)
fig_box_glu.update_layout(showlegend=False)
right.plotly_chart(fig_box_glu, use_container_width=True)

mean_table = (
    plot_df.groupby("뇌졸중 여부")[["age", "avg_glucose_level"]]
    .mean()
    .round(2)
    .reindex(["겪지 않음", "겪음"])
    .rename(columns={"age": "평균 나이", "avg_glucose_level": "평균 혈당의 평균"})
    .reset_index()
)
st.markdown("**두 그룹의 평균값**")
st.dataframe(mean_table, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- 3. 고혈압·심장병별 뇌졸중 비율
st.header("3. 고혈압·심장병 여부에 따른 뇌졸중 비율")
left, right = st.columns(2)

for col_widget, col_name, title in [
    (left, "hypertension", "고혈압 여부별 뇌졸중 비율"),
    (right, "heart_disease", "심장병 여부별 뇌졸중 비율"),
]:
    rate = stroke_rate_by(df, col_name)
    fig = px.bar(
        rate, x="구분", y="뇌졸중 비율(%)", color="구분",
        text="뇌졸중 비율(%)", title=title,
        category_orders={"구분": ["없음", "있음"]},
        hover_data={"사람 수": True},
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(showlegend=False, yaxis_title="뇌졸중 비율(%)", xaxis_title="")
    fig.update_yaxes(range=[0, max(rate["뇌졸중 비율(%)"].max() * 1.25, 1)])
    col_widget.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- 4. bmi 빈 값
st.header("4. bmi가 비어 있는 사람들")

bmi_missing = df[df["bmi"].isna()]
rows = []
for name, part in [("bmi가 빈 사람들", bmi_missing), ("전체", df)]:
    n = len(part)
    n_stroke = int((part["stroke"] == 1).sum())
    rows.append(
        {
            "구분": name,
            "사람 수": n,
            "뇌졸중 사람 수": n_stroke,
            "뇌졸중 비율(%)": round(n_stroke / n * 100, 2) if n > 0 else None,
        }
    )
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- 5. 흡연 상태
st.header("5. 흡연 상태별 사람 수")
smoking = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
st.dataframe(smoking, use_container_width=True, hide_index=True)
