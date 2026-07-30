from io import BytesIO
import re

import numpy as np
import pandas as pd
import plotly.express as px
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
    "19% 미만": "#fff5eb",
    "19% 이상 ~ 23% 미만": "#fdd0a2",
    "23% 이상 ~ 28% 미만": "#fdae6b",
    "28% 이상 ~ 38% 미만": "#e6550d",
    "38% 이상": "#7f2704",
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


def make_sigungu_data(population: pd.DataFrame, geojson: dict) -> tuple[pd.DataFrame, int]:
    """최신 연도의 읍면동 인구를 시군구별로 합쳐 고령화율을 계산합니다."""
    # 연도에 문자열이 섞여 있어도 비교할 수 있도록 숫자로 바꿉니다.
    years = pd.to_numeric(population["연도"], errors="coerce")
    latest_year = int(years.max())
    latest = population.loc[years.eq(latest_year)].copy()

    # '계_65세'부터 '계_100세 이상'까지가 65세 이상 인구입니다.
    total_columns = [column for column in latest.columns if column.startswith("계_")]

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
    numeric = latest[total_columns].replace(",", "", regex=True).apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)
    latest["전체인구"] = numeric.sum(axis=1)
    latest["고령인구"] = numeric[senior_columns].sum(axis=1)
    latest["시군구코드"] = latest["코드"].astype("string").str.zfill(10).str[:5]

    grouped = latest.groupby("시군구코드", as_index=False)[["전체인구", "고령인구"]].sum()
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
    result = areas.merge(grouped, on="시군구코드", how="left")
    result["고령화 단계"] = pd.cut(
        result["고령화율"],
        bins=[-np.inf, 19, 23, 28, 38, np.inf],
        labels=RATE_LABELS,
        right=False,
    )
    return result, latest_year


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
    figure.update_geos(fitbounds="locations", visible=False)
    figure.update_layout(
        height=760,
        margin=dict(l=0, r=0, t=10, b=0),
        legend_title_text="고령화율 구간",
        legend=dict(orientation="v", x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.88)"),
        paper_bgcolor="white",
    )
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
        sigungu_data, year = make_sigungu_data(population_data, boundary_data)

    st.caption(f"{year}년 기준 · 고령화율 = 65세 이상 인구 ÷ 전체 인구 × 100")
    st.plotly_chart(draw_map(sigungu_data, boundary_data), use_container_width=True)

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
