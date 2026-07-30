from io import BytesIO
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


# 스트림릿 페이지의 기본 설정입니다.
st.set_page_config(page_title="전국 고령화 지도", page_icon="🗺️", layout="wide")

POPULATION_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/main/data/"
    "population_yearly.csv.gz"
)
BOUNDARY_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/main/data/"
    "boundaries/sigungu_kr.geojson"
)

# 지도에 표시할 다섯 구간과 색입니다. 순서가 뒤바뀌지 않도록 따로 적어 둡니다.
RATE_LABELS = ["19% 미만", "19% 이상 ~ 23% 미만", "23% 이상 ~ 28% 미만", "28% 이상 ~ 38% 미만", "38% 이상"]
RATE_COLORS = {
    "19% 미만": "#eff3ff",
    "19% 이상 ~ 23% 미만": "#bdd7e7",
    "23% 이상 ~ 28% 미만": "#6baed6",
    "28% 이상 ~ 38% 미만": "#3182bd",
    "38% 이상": "#08519c",
}


@st.cache_data(show_spinner=False)
def load_population() -> pd.DataFrame:
    """압축 인구 자료를 내려받습니다. 코드는 계산하지 않고 글자로 읽습니다."""
    response = requests.get(POPULATION_URL, timeout=60)
    response.raise_for_status()
    return pd.read_csv(BytesIO(response.content), compression="gzip", dtype={"코드": "string"})


@st.cache_data(show_spinner=False)
def load_boundaries() -> dict:
    """시군구 경계 GeoJSON을 내려받습니다."""
    response = requests.get(BOUNDARY_URL, timeout=60)
    response.raise_for_status()
    geojson = response.json()
    # GeoJSON 쪽 코드도 반드시 5자리 글자로 통일해야 지도 색이 정확히 붙습니다.
    for feature in geojson["features"]:
        feature["properties"]["코드"] = str(feature["properties"]["코드"]).zfill(5)
    return geojson


def make_sigungu_data(
    population: pd.DataFrame, geojson: dict
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """모든 연도의 읍면동 인구를 시군구별로 합쳐 고령화율을 계산합니다."""
    # 연도에 문자열이 섞여 있어도 비교할 수 있도록 숫자로 바꿉니다.
    years = pd.to_numeric(population["연도"], errors="coerce")
    latest_year = int(years.max())
    population = population.loc[years.notna()].copy()
    population["연도"] = years.loc[years.notna()].astype(int)

    # '계_65세'부터 '계_100세 이상'까지가 65세 이상 인구입니다.
    total_columns = [column for column in population.columns if column.startswith("계_")]

    def age_of(column: str) -> int | None:
        match = re.fullmatch(r"계_(\d+)세(?: 이상)?", column)
        return int(match.group(1)) if match else None

    senior_columns = [
        column for column in total_columns
        if age_of(column) is not None and age_of(column) >= 65
    ]
    if not total_columns or not senior_columns:
        raise ValueError("나이별 인구 열을 찾지 못했습니다.")

    # 빈칸이나 쉼표가 들어간 값도 안전하게 숫자로 바꿉니다.
    numeric = population[total_columns].replace(",", "", regex=True).apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)
    population["전체인구"] = numeric.sum(axis=1)
    population["고령인구"] = numeric[senior_columns].sum(axis=1)
    population["시군구코드"] = population["코드"].astype("string").str.zfill(10).str[:5]

    grouped = population.groupby(["연도", "시군구코드"], as_index=False)[
        ["전체인구", "고령인구"]
    ].sum()
    grouped["고령화율"] = np.where(
        grouped["전체인구"] > 0,
        grouped["고령인구"] / grouped["전체인구"] * 100,
        np.nan,
    )

    # 명칭은 인구 파일이 아니라 지도 경계의 코드에 붙은 속성을 사용합니다.
    areas = pd.DataFrame(
        {
            "시군구코드": [str(feature["properties"]["코드"]).zfill(5) for feature in geojson["features"]],
            "시군구": [feature["properties"]["시군구"] for feature in geojson["features"]],
            "시도": [feature["properties"]["시도"] for feature in geojson["features"]],
        }
    )
    latest_result = areas.merge(
        grouped.loc[grouped["연도"].eq(latest_year)].drop(columns="연도"),
        on="시군구코드",
        how="left",
    )
    latest_result["고령화 단계"] = pd.cut(
        latest_result["고령화율"],
        bins=[-np.inf, 19, 23, 28, 38, np.inf],
        labels=RATE_LABELS,
        right=False,
    )

    # 애니메이션은 시도별로 합산합니다. 시군구가 분리·통합되어 코드가 바뀌어도
    # 시도 전체 인구에는 빠짐없이 포함되므로 과거 연도의 빈칸을 없앨 수 있습니다.
    population["시도_현재명"] = population["시도"].replace(
        {
            "강원도": "강원특별자치도",
            "전라북도": "전북특별자치도",
        }
    )
    province_grouped = population.groupby(["연도", "시도_현재명"], as_index=False)[
        ["전체인구", "고령인구"]
    ].sum()
    province_grouped["고령화율"] = np.where(
        province_grouped["전체인구"] > 0,
        province_grouped["고령인구"] / province_grouped["전체인구"] * 100,
        np.nan,
    )

    # 현재 시군구 경계 각각에 해당 시도의 동일한 값을 넣어 시도 전체를 칠합니다.
    animation_result = areas.merge(
        province_grouped,
        left_on="시도",
        right_on="시도_현재명",
        how="left",
    ).drop(columns="시도_현재명")
    animation_result["고령화 단계"] = pd.cut(
        animation_result["고령화율"],
        bins=[-np.inf, 19, 23, 28, 38, np.inf],
        labels=RATE_LABELS,
        right=False,
    )
    return latest_result, animation_result, latest_year


def draw_map(data: pd.DataFrame, geojson: dict):
    """배경 타일 없이 시군구 경계와 단계별 색만 그립니다."""
    figure = px.choropleth(
        data,
        geojson=geojson,
        locations="시군구코드",
        featureidkey="properties.코드",
        color="고령화 단계",
        category_orders={"고령화 단계": RATE_LABELS},
        color_discrete_map=RATE_COLORS,
        custom_data=["시군구", "시도", "고령화율"],
    )
    figure.update_traces(
        marker_line_color="#777777",
        marker_line_width=0.45,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "시도: %{customdata[1]}<br>"
            "고령화율: %{customdata[2]:.1f}%<extra></extra>"
        ),
    )

    # 실제 위치를 추가로 찍지 않고, 최고·최저 지역 정보를 범례 항목으로만 보여 줍니다.
    valid_data = data.dropna(subset=["고령화율"])
    if not valid_data.empty:
        highest = valid_data.loc[valid_data["고령화율"].idxmax()]
        lowest = valid_data.loc[valid_data["고령화율"].idxmin()]
        legend_areas = [
            ("최고", highest, RATE_COLORS["38% 이상"]),
            ("최저", lowest, RATE_COLORS["19% 미만"]),
        ]

        jeju_city = valid_data.loc[
            valid_data["시도"].eq("제주특별자치도") & valid_data["시군구"].eq("제주시")
        ]
        if not jeju_city.empty:
            jeju_city = jeju_city.iloc[0]
            legend_areas.append(
                ("제주시", jeju_city, RATE_COLORS[str(jeju_city["고령화 단계"])])
            )

        for label, area, color in legend_areas:
            figure.add_trace(
                go.Scattergeo(
                    lon=[None],
                    lat=[None],
                    mode="markers",
                    marker=dict(size=10, color=color, line=dict(color="#555555", width=0.5)),
                    name=(
                        f"{label}: {area['시도']} {area['시군구']} "
                        f"({area['고령화율']:.1f}%)"
                    ),
                    hoverinfo="skip",
                    showlegend=True,
                )
            )

    figure.update_geos(fitbounds="locations", visible=False)
    figure.update_layout(
        height=760,
        margin=dict(l=0, r=0, t=10, b=0),
        legend_title_text="고령화율 구간",
        legend=dict(orientation="v", x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.88)"),
        paper_bgcolor="white",
    )
    return figure


def draw_animated_map(data: pd.DataFrame, geojson: dict):
    """연도 슬라이더와 재생 버튼이 있는 시도별 고령화율 지도를 만듭니다."""
    hover_template = (
        "<b>%{customdata[0]}</b><br>"
        "연도: %{customdata[2]}년<br>"
        "고령화율: %{customdata[1]:.1f}%<extra></extra>"
    )

    # 범주형 색상은 연도마다 트레이스 수가 달라져 애니메이션 중 일부 지역이
    # 사라질 수 있습니다. 하나의 숫자 트레이스를 쓰되 경계에서 색을 끊어
    # 겉보기에는 정확히 다섯 단계가 되도록 색상표를 만듭니다.
    color_max = max(45.0, float(np.ceil(data["고령화율"].max())))
    boundaries = [19 / color_max, 23 / color_max, 28 / color_max, 38 / color_max]
    colors = [RATE_COLORS[label] for label in RATE_LABELS]
    step_color_scale = [
        [0.0, colors[0]],
        [boundaries[0], colors[0]],
        [boundaries[0], colors[1]],
        [boundaries[1], colors[1]],
        [boundaries[1], colors[2]],
        [boundaries[2], colors[2]],
        [boundaries[2], colors[3]],
        [boundaries[3], colors[3]],
        [boundaries[3], colors[4]],
        [1.0, colors[4]],
    ]
    figure = px.choropleth(
        data.sort_values("연도"),
        geojson=geojson,
        locations="시군구코드",
        featureidkey="properties.코드",
        color="고령화율",
        animation_frame="연도",
        color_continuous_scale=step_color_scale,
        range_color=(0, color_max),
        custom_data=["시도", "고령화율", "연도"],
    )
    figure.update_traces(
        marker_line_color="#777777",
        marker_line_width=0.45,
        hovertemplate=hover_template,
    )
    # 애니메이션의 각 연도 프레임에도 같은 경계선과 설명 형식을 적용합니다.
    for frame in figure.frames:
        for trace in frame.data:
            trace.update(
                marker_line_color="#777777",
                marker_line_width=0.45,
                hovertemplate=hover_template,
            )
    figure.update_geos(fitbounds="locations", visible=False)
    figure.update_layout(
        height=800,
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar=dict(
            title="고령화율 구간",
            tickmode="array",
            tickvals=[9.5, 21, 25.5, 33, (38 + color_max) / 2],
            ticktext=RATE_LABELS,
            len=0.55,
        ),
        paper_bgcolor="white",
    )

    # 재생 속도를 너무 빠르지 않게 조정해 연도별 변화를 알아보기 쉽게 합니다.
    if figure.layout.updatemenus:
        play_button = figure.layout.updatemenus[0].buttons[0]
        play_button.args[1]["frame"]["duration"] = 800
        play_button.args[1]["transition"]["duration"] = 300
    return figure


def ranking_table(data: pd.DataFrame, ascending: bool) -> pd.DataFrame:
    """고령화율 상위 또는 하위 10곳을 보기 좋은 표로 만듭니다."""
    ranked = (
        data.dropna(subset=["고령화율"])
        .sort_values("고령화율", ascending=ascending)
        .head(10)[["시도", "시군구", "고령화율"]]
        .copy()
    )
    ranked.insert(0, "순위", range(1, len(ranked) + 1))
    ranked["고령화율"] = ranked["고령화율"].map(lambda value: f"{value:.1f}%")
    return ranked


st.title("전국 시군구 고령화 지도")

try:
    with st.spinner("최신 인구와 지도 경계를 불러오는 중입니다..."):
        population_data = load_population()
        boundary_data = load_boundaries()
        sigungu_data, yearly_data, year = make_sigungu_data(population_data, boundary_data)

    # 화면이 복잡해지지 않도록 핵심 요약 정보는 사이드바에도 모아 보여 줍니다.
    valid_sigungu = sigungu_data.dropna(subset=["고령화율"])
    highest_area = valid_sigungu.loc[valid_sigungu["고령화율"].idxmax()]
    lowest_area = valid_sigungu.loc[valid_sigungu["고령화율"].idxmin()]
    jeju_area = valid_sigungu.loc[
        valid_sigungu["시도"].eq("제주특별자치도")
        & valid_sigungu["시군구"].eq("제주시")
    ]

    with st.sidebar:
        st.header("지도 정보")
        st.metric("기준 연도", f"{year}년")
        st.caption("고령화율 = 65세 이상 인구 ÷ 전체 인구 × 100")
        st.divider()
        st.subheader("시군구 요약")
        st.metric(
            "고령화율 최고",
            f"{highest_area['고령화율']:.1f}%",
            help=f"{highest_area['시도']} {highest_area['시군구']}",
        )
        st.caption(f"{highest_area['시도']} {highest_area['시군구']}")
        st.metric(
            "고령화율 최저",
            f"{lowest_area['고령화율']:.1f}%",
            help=f"{lowest_area['시도']} {lowest_area['시군구']}",
        )
        st.caption(f"{lowest_area['시도']} {lowest_area['시군구']}")
        if not jeju_area.empty:
            jeju_area = jeju_area.iloc[0]
            st.metric("제주특별자치도 제주시", f"{jeju_area['고령화율']:.1f}%")
        st.divider()
        st.subheader("애니메이션 안내")
        st.caption("애니메이션은 17개 시도 기준입니다. 재생 버튼이나 연도 슬라이더로 변화를 확인하세요.")

    st.caption(f"{year}년 기준 · 고령화율 = 65세 이상 인구 ÷ 전체 인구 × 100")
    latest_tab, animation_tab = st.tabs([f"{year}년 시군구 지도", "연도별 시도 변화 애니메이션"])
    with latest_tab:
        st.plotly_chart(draw_map(sigungu_data, boundary_data), use_container_width=True)
    with animation_tab:
        st.caption("17개 시도 기준입니다. 재생 버튼을 누르거나 연도 슬라이더를 움직여 보세요.")
        st.plotly_chart(draw_animated_map(yearly_data, boundary_data), use_container_width=True)

    st.subheader("시군구 고령화율 순위")
    high_column, low_column = st.columns(2)
    with high_column:
        st.markdown("#### 높은 곳 10개")
        st.dataframe(ranking_table(sigungu_data, ascending=False), hide_index=True, use_container_width=True)
    with low_column:
        st.markdown("#### 낮은 곳 10개")
        st.dataframe(ranking_table(sigungu_data, ascending=True), hide_index=True, use_container_width=True)

except (requests.RequestException, ValueError, KeyError, pd.errors.ParserError) as error:
    st.error(f"데이터를 불러오거나 처리하지 못했습니다: {error}")
