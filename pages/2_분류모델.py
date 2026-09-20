import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

st.set_page_config(page_title="분류 모델 - 뇌졸중 예측 실습실", page_icon="🤖", layout="wide")

# ---------------------------------------------------------------- 이름과 색
FEATURES = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
DEFAULT_FEATURES = [f for f in FEATURES if f != "bmi"]
KOR = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
JOSA = {  # 주어 뒤에 붙는 '가/이'
    "age": "가",
    "avg_glucose_level": "이",
    "bmi": "가",
    "hypertension": "이",
    "heart_disease": "이",
}
BINARY = {"hypertension", "heart_disease"}  # 0 또는 1만 가지는 속성

LOGI = "로지스틱 회귀(확률로 답하는 모델)"
TREE = "의사결정트리(질문으로 답하는 모델)"
BASE = "입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델"

NEG_COLOR = "#4c78a8"  # 뇌졸중 아님(0)
POS_COLOR = "#e4572e"  # 뇌졸중(1) = 양성
NEG_TINT = "rgba(76, 120, 168, 0.16)"
POS_TINT = "rgba(228, 87, 46, 0.22)"

st.markdown(
    """
    <style>
    .acc-card { border: 1px solid rgba(128,128,128,0.35); border-radius: 14px;
                padding: 1rem 1.2rem; height: 100%; }
    .acc-title { font-size: 0.95rem; font-weight: 600; min-height: 2.8em; opacity: 0.9; }
    .acc-label { font-size: 0.8rem; opacity: 0.65; margin-top: 0.4rem; }
    .acc-big { font-size: 3rem; font-weight: 700; line-height: 1.15; }
    .acc-small { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;
                 font-size: 0.85rem; opacity: 0.75; margin-top: 0.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def accuracy_card(title: str, train_acc: float, test_acc: float) -> str:
    return (
        '<div class="acc-card">'
        f'<div class="acc-title">{title}</div>'
        '<div class="acc-label">테스트 데이터 정확도</div>'
        f'<div class="acc-big">{pct(test_acc)}</div>'
        '<div class="acc-small">'
        f"<span>훈련 데이터 {pct(train_acc)}</span>"
        f"<span>테스트 데이터 {pct(test_acc)}</span>"
        "</div></div>"
    )


def view_range(col: str, values: pd.Series) -> tuple:
    """그림에서 보여 줄 축의 범위."""
    if col in BINARY:
        return -0.3, 1.3
    lo, hi = float(values.min()), float(values.max())
    pad = (hi - lo) * 0.05 or 0.5
    return lo - pad, hi + pad


def clip_line(a: float, b: float, c: float, x0: float, x1: float, y0: float, y1: float) -> list:
    """직선 a*x + b*y + c = 0 이 사각형 [x0,x1]x[y0,y1] 안에서 지나는 점들."""
    pts = []
    if b != 0:
        for x in (x0, x1):
            y = -(a * x + c) / b
            if y0 <= y <= y1:
                pts.append((x, y))
    if a != 0:
        for y in (y0, y1):
            x = -(b * y + c) / a
            if x0 <= x <= x1:
                pts.append((x, y))
    pts.sort()
    return pts


def build_dot(model: DecisionTreeClassifier, features: list, n_node, n_pos) -> str:
    """의사결정트리를 DOT 문자열로 바꾼다 (st.graphviz_chart에 넘길 용도)."""
    t = model.tree_
    font = "Malgun Gothic, Apple SD Gothic Neo, Noto Sans KR, sans-serif"
    out = [
        "digraph Tree {",
        'graph [bgcolor="white", rankdir=TB, nodesep=0.3, ranksep=0.6];',
        f'node [shape=box, style="rounded,filled", fontname="{font}", fontsize=12, '
        'fontcolor="#222222", color="#666666", margin="0.2,0.12"];',
        f'edge [fontname="{font}", fontsize=12, fontcolor="#222222", color="#666666"];',
    ]
    for i in range(t.node_count):
        n, p = int(n_node[i]), int(n_pos[i])
        ratio = p / n * 100 if n else 0.0
        stat = f"훈련용 {n:,}명 중 뇌졸중 {p:,}명 ({ratio:.1f}%)"
        left, right = int(t.children_left[i]), int(t.children_right[i])
        if left == -1:  # 더 묻지 않고 답을 내는 마디
            answer = model.classes_[t.value[i][0].argmax()]
            if answer == 1:
                title, fill = "답: 뇌졸중", "#f6c9c0"
            else:
                title, fill = "답: 뇌졸중 아님", "#cfe3f7"
        else:  # 질문을 던지는 마디
            feat = features[int(t.feature[i])]
            if feat in BINARY:
                title = f"{KOR[feat]}{JOSA[feat]} 없는가?"
            else:
                title = f"{KOR[feat]}{JOSA[feat]} {t.threshold[i]:.4g} 이하인가?"
            fill = "#f2f2f2"
        out.append(f'n{i} [label="{title}\\n{stat}", fillcolor="{fill}"];')
        if left != -1:
            out.append(f'n{i} -> n{left} [label="예"];')
            out.append(f'n{i} -> n{right} [label="아니요"];')
    out.append("}")
    return "\n".join(out)


# ---------------------------------------------------------------- 화면 맨 위
st.title("🤖 분류 모델")
st.caption("뇌졸중(stroke가 1)을 양성으로 두고, 두 가지 모델로 뇌졸중 여부를 맞혀 봅니다.")

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다. 인터넷 연결과 주소를 확인해 주세요.\n\n{e}")
    st.stop()

# ---------------------------------------------------------------- 1. 입력 속성 고르기
st.header("1. 입력으로 사용할 속성 고르기")
chosen = st.multiselect(
    "입력 속성",
    options=FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda c: KOR[c],
    key="chosen_features",
)
features = [f for f in FEATURES if f in chosen]  # 항상 같은 순서로 정리

if len(features) < 2:
    st.info("입력 속성을 두 개 이상 골라 주세요. 두 개보다 적으면 모델과 그림을 보여 드릴 수 없어요.")
    st.stop()

# ---------------------------------------------------------------- 2. 훈련용/테스트용 나누기
data = df.sort_values("id").reset_index(drop=True)
is_test = (data.index % 10) < 3  # 번호 순 열 명 묶음마다 앞 세 명 = 테스트용
test_df = data[is_test]
train_df = data[~is_test]

X_train = train_df[features].copy()
X_test = test_df[features].copy()
y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

st.caption(
    f"번호(id) 순으로 정렬해 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 고정했습니다. "
    f"테스트용 {len(test_df):,}명, 나머지 훈련용 {len(train_df):,}명입니다."
)

if "bmi" in features:
    bmi_median = float(X_train["bmi"].median())
    n_fill_train = int(X_train["bmi"].isna().sum())
    n_fill_test = int(X_test["bmi"].isna().sum())
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)
    st.caption(
        f"체질량지수가 비어 있던 사람(훈련용 {n_fill_train:,}명, 테스트용 {n_fill_test:,}명)은 "
        f"훈련용의 중앙값 {bmi_median:.4g}으로 채웠습니다. 그림에서도 이 채운 값으로 찍힙니다."
    )

# ---------------------------------------------------------------- 3. 모델 만들기
# 크기 맞추기(표준화)는 훈련용으로만 맞추고, 로지스틱 회귀에 적용한다.
logi = Pipeline(
    [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]
)
logi.fit(X_train, y_train)

# 의사결정트리는 크기를 맞추지 않아도 같은 나눔이 나오므로 원래 값 그대로 학습한다.
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
tree.fit(X_train, y_train)

base = DummyClassifier(strategy="most_frequent")
base.fit(X_train, y_train)

# ---------------------------------------------------------------- 4. 정확도 카드
st.header("2. 모델의 정확도")
cards = [
    (LOGI, logi),
    (TREE, tree),
    (BASE, base),
]
cols = st.columns(3)
for col, (name, model) in zip(cols, cards):
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    col.markdown(accuracy_card(name, train_acc, test_acc), unsafe_allow_html=True)

st.caption(
    "크기 맞추기는 훈련용 사람들로만 맞춰서 "
    f"{LOGI}에 사용했습니다. {TREE}는 크기를 맞추지 않아도 같은 나눔이 나옵니다. "
    "카드의 큰 숫자는 테스트 데이터의 정확도입니다."
)

# ---------------------------------------------------------------- 5. 산점도
st.header("3. 두 속성으로 보는 테스트 데이터")

sel_x, sel_y = st.columns(2)
fx = sel_x.selectbox("가로축", features, index=0, format_func=lambda c: KOR[c], key="axis_x")
y_options = [f for f in features if f != fx]
fy = sel_y.selectbox("세로축", y_options, index=0, format_func=lambda c: KOR[c], key="axis_y")

medians = X_test.median()  # 테스트 데이터의 중앙값
others = [f for f in features if f not in (fx, fy)]


def make_frame(points: list) -> pd.DataFrame:
    """(가로축 값, 세로축 값) 목록으로, 나머지 속성은 중앙값에 세운 표를 만든다."""
    d = {f: [float(medians[f])] * len(points) for f in features}
    d[fx] = [p[0] for p in points]
    d[fy] = [p[1] for p in points]
    return pd.DataFrame(d)[features]


x0, x1 = view_range(fx, X_test[fx])
y0, y1 = view_range(fy, X_test[fy])

# 의사결정트리 모델이 나눈 칸 (그림 전체를 격자로 나누어 답을 물어봄)
N = 150
xs = [x0 + (x1 - x0) * i / (N - 1) for i in range(N)]
ys = [y0 + (y1 - y0) * j / (N - 1) for j in range(N)]
grid = make_frame([(x, y) for y in ys for x in xs])
tree_pred = tree.predict(grid).tolist()
z_rows = [tree_pred[j * N:(j + 1) * N] for j in range(N)]

fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        x=xs, y=ys, z=z_rows, zmin=0, zmax=1,
        colorscale=[[0, NEG_TINT], [1, POS_TINT]],
        showscale=False, hoverinfo="skip",
    )
)

# 테스트 데이터 점 (뇌졸중이 위에 보이도록 나중에 그림)
plot_df = X_test.assign(stroke=y_test)
for value, label, color, size in [
    (0, "실제 뇌졸중 아님(0)", NEG_COLOR, 6),
    (1, "실제 뇌졸중(1)", POS_COLOR, 8),
]:
    sub = plot_df[plot_df["stroke"] == value]
    fig.add_trace(
        go.Scatter(
            x=sub[fx], y=sub[fy], mode="markers", name=label,
            marker=dict(color=color, size=size, opacity=0.65, line=dict(width=0.5, color="white")),
            hovertemplate=f"{KOR[fx]}: %{{x}}<br>{KOR[fy]}: %{{y}}<extra>{label}</extra>",
        )
    )

# 로지스틱 회귀가 뇌졸중 확률을 0.5로 가르는 선: a*x + b*y + c = 0
pos_idx = list(logi.named_steps["model"].classes_).index(1)
z3 = logi.decision_function(make_frame([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]))
c_ = float(z3[0])
a_ = float(z3[1]) - c_
b_ = float(z3[2]) - c_

corner_probs = logi.predict_proba(make_frame([(x0, y0), (x1, y0), (x0, y1), (x1, y1)]))[:, pos_idx].tolist()
p_min, p_max = min(corner_probs), max(corner_probs)
line_in_view = p_min < 0.5 < p_max

if line_in_view:
    pts = clip_line(a_, b_, c_, x0, x1, y0, y1)
    if len(pts) >= 2:
        fig.add_trace(
            go.Scatter(
                x=[pts[0][0], pts[-1][0]], y=[pts[0][1], pts[-1][1]], mode="lines",
                name=f"{LOGI}의 뇌졸중 확률 0.5 경계선",
                line=dict(color="black", width=3), hoverinfo="skip",
            )
        )

# 의사결정트리 칸의 색 설명 (그림에 실제로 나온 색만 범례에 넣음)
if 0 in tree_pred:
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="square", size=14, color=NEG_TINT, line=dict(width=1, color=NEG_COLOR)),
            name=f"{TREE}가 '뇌졸중 아님'이라고 답하는 칸",
        )
    )
if 1 in tree_pred:
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="square", size=14, color=POS_TINT, line=dict(width=1, color=POS_COLOR)),
            name=f"{TREE}가 '뇌졸중'이라고 답하는 칸",
        )
    )

fig.update_layout(
    height=620,
    xaxis_title=KOR[fx],
    yaxis_title=KOR[fy],
    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
    margin=dict(l=10, r=10, t=20, b=110),
)
fig.update_xaxes(range=[x0, x1])
fig.update_yaxes(range=[y0, y1])
if fx in BINARY:
    fig.update_xaxes(tickvals=[0, 1], ticktext=["없음(0)", "있음(1)"])
if fy in BINARY:
    fig.update_yaxes(tickvals=[0, 1], ticktext=["없음(0)", "있음(1)"])

st.plotly_chart(fig)

st.caption(f"테스트 데이터 {len(test_df):,}명을 점으로 찍었습니다. 가로축은 {KOR[fx]}, 세로축은 {KOR[fy]}입니다.")

if others:
    fixed_text = ", ".join(f"{KOR[f]} = {medians[f]:.4g}" for f in others)
    st.write(f"그림에 나오지 않은 속성은 테스트 데이터의 중앙값에 세워 두고 계산했습니다: {fixed_text}")
else:
    st.write("고른 속성이 이 두 개뿐이라서, 따로 중앙값에 세워 둔 속성은 없습니다.")

if line_in_view:
    st.write(f"검은 선은 {LOGI}가 뇌졸중 확률을 0.5로 가르는 자리입니다.")
elif p_max <= 0.5:
    st.write(
        f"{LOGI}의 0.5 경계선은 이 그림 밖에 있습니다. "
        f"그림 안의 뇌졸중 확률은 {pct(p_min)}~{pct(p_max)}로 어디서도 0.5에 닿지 않아, "
        "그림 안의 모든 자리를 '뇌졸중 아님'이라고 답합니다."
    )
else:
    st.write(
        f"{LOGI}의 0.5 경계선은 이 그림 밖에 있습니다. "
        f"그림 안의 뇌졸중 확률은 {pct(p_min)}~{pct(p_max)}로 모두 0.5보다 높아, "
        "그림 안의 모든 자리를 '뇌졸중'이라고 답합니다."
    )

# ---------------------------------------------------------------- 6. 의사결정트리 가지 그림
st.header(f"4. {TREE}가 던진 질문")

path = tree.decision_path(X_train)
n_node = path.T.dot(pd.Series(1, index=y_train.index).to_numpy())  # 마디마다 훈련용 사람 수
n_pos = path.T.dot(y_train.to_numpy())  # 마디마다 실제 뇌졸중인 훈련용 사람 수

st.graphviz_chart(build_dot(tree, features, n_node, n_pos))
st.caption("예는 왼쪽 가지, 아니요는 오른쪽 가지입니다. 파란 마디는 '뇌졸중 아님', 붉은 마디는 '뇌졸중'이라고 답하는 마디입니다.")

t = tree.tree_
leaf_answers = [
    tree.classes_[t.value[i][0].argmax()]
    for i in range(t.node_count)
    if t.children_left[i] == -1
]
n_leaves = len(leaf_answers)
n_no = sum(1 for a in leaf_answers if a == 0)
asked = {int(k) for k in t.feature if k >= 0}  # 질문에 쓰인 속성의 번호
used = [KOR[f] for k, f in enumerate(features) if k in asked]

st.write(f"답을 내는 마디는 모두 {n_leaves}칸이고, 그중 {n_no}칸이 '뇌졸중 아님'이라고 답합니다.")
if used:
    st.write(f"고른 속성 가운데 이 나무가 실제로 물은 것은 {', '.join(used)}입니다.")
else:
    st.write("고른 속성 가운데 이 나무가 실제로 물은 것은 없습니다. 질문을 하나도 던지지 않았어요.")
