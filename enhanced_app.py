from typing import Optional, List
import io
import logging
from datetime import date, timedelta
import xml.etree.ElementTree as ET

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("covid_dashboard")

# ---------- Page config & CSS ----------
st.set_page_config(page_title="COVID-19 Dashboard", page_icon="🦠", layout="wide")

APP_CSS = """
<style>
body {
    background-color: #121212;
    color: #ffffff;
}
.header {
    font-size: 28px;
    font-weight: 700;
    color: #4da3ff;
}
.section {
    font-size: 16px;
    font-weight: 600;
    color: #dddddd;
    border-left: 4px solid #4da3ff;
    padding-left: 10px;
    margin-top: 12px;
    margin-bottom: 8px;
}
.kpi {
    font-size: 18px;
    color: #ffffff;
}
.small-muted { color: #aaaaaa; font-size:12px; }
.stApp {
    background-color: #121212;
}
.news-item {
    background-color: #1e1e1e;
    padding: 10px;
    margin: 5px 0;
    border-radius: 5px;
}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

# ---------- Constants ----------
OWID_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
LOCAL_CSV_PATH = "data/owid-covid-data.csv"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q=COVID-19+when:7d&hl=en-US&gl=US&ceid=US:en"
DEFAULT_COUNTRY_SELECTION = ["World", "India", "United States", "China", "Brazil"]
COLOR_SCALE = "Turbo"  # Deeper, more distinct colors
LINE_PALETTE = px.colors.qualitative.Set2
MAX_COUNTRIES = 8
PLOT_TEMPLATE = "plotly_dark"  # Dark theme for plots

# ---------- Helper Functions ----------
@st.cache_data(show_spinner="Loading local data...")
def load_csv_from_path(path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        logger.info("Loaded CSV from path: %s", path)
        return df
    except FileNotFoundError:
        logger.info("CSV not found at path: %s", path)
        return None
    except Exception as e:
        logger.exception("Error loading CSV from path: %s", e)
        return None

@st.cache_data(show_spinner="Fetching latest OWID data...")
def fetch_owid_data() -> Optional[pd.DataFrame]:
    try:
        response = requests.get(OWID_URL, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(io.BytesIO(response.content), parse_dates=["date"])
        logger.info("Fetched latest OWID data from %s", OWID_URL)
        return df
    except Exception as e:
        logger.exception("Error fetching OWID data: %s", e)
        return None

@st.cache_data(show_spinner=False)
def read_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes), parse_dates=["date"])
    return df

@st.cache_data(show_spinner="Fetching latest COVID news...")
def fetch_covid_news() -> List[dict]:
    try:
        response = requests.get(GOOGLE_NEWS_RSS, timeout=30)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        items = []
        for item in root.findall('.//item')[:10]:  # Top 10 recent news
            title = item.find('title').text if item.find('title') is not None else ''
            link = item.find('link').text if item.find('link') is not None else ''
            pubDate = item.find('pubDate').text if item.find('pubDate') is not None else ''
            description = item.find('description').text if item.find('description') is not None else ''
            items.append({'title': title, 'link': link, 'pubDate': pubDate, 'description': description})
        return items
    except Exception as e:
        logger.exception("Error fetching news: %s", e)
        return []

def validate_df(df: pd.DataFrame) -> bool:
    required_cols = {"date", "location"}
    return required_cols.issubset(set(df.columns))

def safe_latest_by_location(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("date").groupby("location", as_index=False).last()

def format_value(v):
    if pd.isna(v):
        return "N/A"
    if abs(v) >= 1_000_000_000:
        return f"{v/1_000_000_000:,.2f}B"
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:,.2f}M"
    return f"{int(v):,}"

# ---------- Data Ingestion ----------
st.markdown('<div class="header">🌍 COVID-19 Analytics</div>', unsafe_allow_html=True)
st.markdown("**Source:** Our World in Data (OWID) COVID-19 dataset. Auto-fetch latest if no local/upload.")

col_upload, col_info = st.columns([2, 3])

with col_upload:
    uploaded_file = st.file_uploader("Upload CSV (optional)", type=["csv"], help="Upload owid-covid-data.csv or compatible.")
    use_local = False
    use_fetch = False
    local_df = load_csv_from_path(LOCAL_CSV_PATH)
    if local_df is not None:
        use_local = st.checkbox(f"Use local file: {LOCAL_CSV_PATH}", value=True)
    else:
        st.info("No local CSV found. You can upload or fetch latest from OWID.")
    
    if not use_local and not uploaded_file:
        use_fetch = st.checkbox("Fetch latest data from OWID", value=True)

with col_info:
    st.markdown("**Quick actions**")
    st.markdown("- Upload, use local, or fetch latest\n- Multi-select countries (up to 8)\n- Download filtered data\n- Toggle log scale or animation in charts")

# Load DataFrame based on priority: upload > local > fetch
df = None
if uploaded_file:
    try:
        df = read_uploaded_csv(uploaded_file.read())
        st.success("Uploaded file loaded successfully.")
    except Exception as e:
        st.error(f"Failed to read uploaded file: {str(e)}")
        st.stop()
elif use_local and local_df is not None:
    df = local_df
    st.success("Local file loaded successfully.")
elif use_fetch:
    df = fetch_owid_data()
    if df is not None:
        st.success("Latest OWID data fetched successfully.")
    else:
        st.error("Failed to fetch OWID data. Please upload or place local CSV.")
        st.stop()
else:
    st.warning("No dataset selected. Enable local/fetch or upload CSV.")
    st.stop()

# Validate and prepare data
if not validate_df(df):
    st.error("Dataset missing required columns ('date', 'location'). Use OWID standard CSV.")
    st.stop()

if not np.issubdtype(df["date"].dtype, np.datetime64):
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

# ---------- Sidebar Controls ----------
with st.sidebar:
    st.header("Filters")
    metric_map = {
        "Total Cases": "total_cases",
        "Total Cases per Million": "total_cases_per_million",
        "Total Deaths": "total_deaths",
        "Total Deaths per Million": "total_deaths_per_million",
        "New Cases (daily)": "new_cases",
        "New Cases per Million": "new_cases_per_million",
        "New Deaths (daily)": "new_deaths",
        "New Deaths per Million": "new_deaths_per_million",
        "Total Vaccinations": "total_vaccinations",
        "People Vaccinated per Hundred": "people_vaccinated_per_hundred",
        "Reproduction Rate": "reproduction_rate",
        "ICU Patients": "icu_patients",
        "Hospital Patients": "hosp_patients",
    }
    selected_metric_label = st.selectbox("Metric", list(metric_map.keys()), index=0)
    selected_metric = metric_map[selected_metric_label]

    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    default_start = max(min_date, max_date - timedelta(days=365))  # Last 1 year
    start_date = st.date_input("Start date", default_start, min_value=min_date, max_value=max_date)
    end_date = st.date_input("End date", max_date, min_value=min_date, max_value=max_date)

    countries = sorted(df["location"].unique())
    selected_countries = st.multiselect("Countries to compare", options=countries, default=DEFAULT_COUNTRY_SELECTION[:MAX_COUNTRIES], help=f"Select up to {MAX_COUNTRIES} for clarity.")

    use_log_scale = st.checkbox("Use logarithmic scale for charts", value=False)
    animate_map = st.checkbox("Animate Map over Time (Monthly)", value=False)
    map_projection = st.selectbox("Map Projection", ["equirectangular", "orthographic"], index=0, help="Equirectangular for smoother rotation, Orthographic for 3D globe.")

# ---------- Filter DataFrame ----------
start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date)
filtered = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].copy()

if filtered.empty:
    st.error("No data for selected date range. Adjust filters.")
    st.stop()

# ---------- Main App: Tabs ----------
tabs = st.tabs(["Overview", "Country Comparison", "Vaccination Focus", "Data Explorer", "News"])
tab_overview, tab_compare, tab_vax, tab_data, tab_news = tabs

# ---------- Overview Tab ----------
# ---------- Overview Tab ----------
with tab_overview:
    st.markdown('<div class="section">Overview</div>', unsafe_allow_html=True)
    latest_by_loc = safe_latest_by_location(filtered)
    world_row = latest_by_loc[latest_by_loc["location"] == "World"]
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown('<div class="kpi"> Total Cases (World)</div>', unsafe_allow_html=True)
        st.markdown(f"**{format_value(world_row['total_cases'].iloc[0] if not world_row.empty else np.nan)}**")
    with c2:
        st.markdown('<div class="kpi"> Total Deaths (World)</div>', unsafe_allow_html=True)
        st.markdown(f"**{format_value(world_row['total_deaths'].iloc[0] if not world_row.empty else np.nan)}**")
    with c3:
        st.markdown('<div class="kpi"> Total Vaccinations (World)</div>', unsafe_allow_html=True)
        st.markdown(f"**{format_value(world_row['total_vaccinations'].iloc[0] if not world_row.empty else np.nan)}**")
    with c4:
        st.markdown('<div class="kpi"> Date Range</div>', unsafe_allow_html=True)
        st.markdown(f"**{start_date} → {end_date}**")

    st.markdown("---")

    st.markdown('<div class="section">Global Map</div>', unsafe_allow_html=True)
    if "iso_code" in filtered.columns and selected_metric in filtered.columns:
        if animate_map:
            # Prepare monthly data for animation
            filtered_monthly = filtered.set_index('date').groupby(['location', pd.Grouper(freq='M')]).last().reset_index()
            filtered_monthly['date_str'] = filtered_monthly['date'].dt.strftime('%Y-%m')
            map_df = filtered_monthly.dropna(subset=[selected_metric, "iso_code"])
            animation_frame = "date_str"
        else:
            map_df = latest_by_loc.dropna(subset=[selected_metric, "iso_code"])
            animation_frame = None

        if not map_df.empty:
            # Calculate percentile-based range for better color distribution
            valid_data = map_df[selected_metric].dropna()
            if len(valid_data) > 0:
                min_val = valid_data.quantile(0.05)  # 5th percentile
                max_val = valid_data.quantile(0.95)  # 95th percentile
                # Add slider in sidebar for custom range adjustment
                with st.sidebar:
                    color_range = st.slider(
                        "Color Range (Min-Max)",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=(float(min_val), float(max_val)),
                        step=(max_val - min_val) / 100
                    )
            else:
                color_range = [valid_data.min(), valid_data.max()] if not valid_data.empty else [0, 1]

            fig = px.choropleth(
                map_df,
                locations="iso_code",
                color=selected_metric,
                hover_name="location",
                title=f"{selected_metric_label} — {'Evolution (Monthly)' if animate_map else 'Latest Data'}",
                color_continuous_scale=COLOR_SCALE,
                range_color=color_range,  # Use adjusted range
                labels={selected_metric: selected_metric_label},
                animation_frame=animation_frame,
            )
            # Enhance with selected projection and dark theme
            fig.update_layout(
                template=PLOT_TEMPLATE,
                height=520,
                margin={"t": 50, "l": 10, "r": 10, "b": 10},
                coloraxis_colorbar_title_side="right",
                coloraxis_colorbar_title_text=selected_metric_label,
                geo=dict(
                    projection_type=map_projection,
                    showocean=True,
                    oceancolor="rgb(20, 40, 60)",
                    showland=True,
                    landcolor="rgb(50, 50, 50)",
                    bgcolor="rgb(18,18,18)",
                    showcountries=True,
                    countrycolor="rgb(80,80,80)",
                    lataxis_range=[-90, 90] if map_projection == "orthographic" else None,
                    lonaxis_range=[-180, 180] if map_projection == "orthographic" else None,
                ),
            )
            if use_log_scale:
                fig.update_layout(coloraxis_type="log")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No geo data for selected metric/date range.")
    else:
        st.warning("No 'iso_code' or selected metric in dataset. Choropleth skipped.")

    st.markdown("---")

    st.markdown('<div class="section">Top 10 Countries</div>', unsafe_allow_html=True)
    exclude_regions = ["World", "Asia", "Europe", "Africa", "North America", "South America", "Oceania", "European Union"]
    candidates = latest_by_loc[~latest_by_loc["location"].isin(exclude_regions)]
    if selected_metric in candidates.columns:
        candidates = candidates.dropna(subset=[selected_metric])
        if not candidates.empty:
            top10 = candidates.nlargest(10, selected_metric)
            fig_bar = px.bar(top10, x="location", y=selected_metric, title=f"Top 10 — {selected_metric_label}", color=selected_metric, color_continuous_scale=COLOR_SCALE)
            fig_bar.update_layout(template=PLOT_TEMPLATE)
            if use_log_scale:
                fig_bar.update_layout(yaxis_type="log")
            fig_bar.update_layout(height=420, margin={"t": 40, "l": 10, "r": 10, "b": 10}, xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No data for top 10 computation.")
    else:
        st.info("Selected metric not available in dataset.")
# ---------- Country Comparison Tab ----------
with tab_compare:
    st.markdown('<div class="section">Country Comparison</div>', unsafe_allow_html=True)
    if not selected_countries:
        st.info("Select countries in sidebar to compare.")
    else:
        selected_countries_trim = selected_countries[:MAX_COUNTRIES]
        if len(selected_countries) > MAX_COUNTRIES:
            st.warning(f"Limited to first {MAX_COUNTRIES} countries for chart clarity.")
        
        # Time Series Line Chart
        fig_line = go.Figure()
        latest_values = {}
        for i, country in enumerate(selected_countries_trim):
            cdf = filtered[filtered["location"] == country]
            if cdf.empty or selected_metric not in cdf.columns or cdf[selected_metric].dropna().empty:
                continue
            fig_line.add_trace(go.Scatter(
                x=cdf["date"],
                y=cdf[selected_metric],
                name=country,
                mode="lines",
                line={"width": 2.5, "color": LINE_PALETTE[i % len(LINE_PALETTE)]}
            ))
            # Store latest for pie
            latest = cdf.iloc[-1][selected_metric] if not pd.isna(cdf.iloc[-1][selected_metric]) else 0
            latest_values[country] = latest
        fig_line.update_layout(
            template=PLOT_TEMPLATE,
            title=f"{selected_metric_label} — Time Series",
            xaxis_title="Date",
            yaxis_title=selected_metric_label,
            height=520,
            legend={"orientation": "h", "y": -0.2}
        )
        if use_log_scale:
            fig_line.update_layout(yaxis_type="log")
        st.plotly_chart(fig_line, use_container_width=True)

        # Additional Visualization: Pie Chart for Proportions
        st.markdown("---")
        st.markdown('<div class="section">Proportions Among Selected Countries</div>', unsafe_allow_html=True)
        if latest_values:
            pie_df = pd.DataFrame(list(latest_values.items()), columns=['Country', selected_metric_label])
            pie_df[selected_metric_label] = pie_df[selected_metric_label].clip(lower=0)  # Ensure non-negative
            fig_pie = px.pie(pie_df, values=selected_metric_label, names='Country', title=f"{selected_metric_label} Distribution")
            fig_pie.update_layout(template=PLOT_TEMPLATE, height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

        # Additional Visualization: Box Plot for Daily Variations (if new_cases available)
        st.markdown("---")
        st.markdown('<div class="section">Daily Variations (Box Plot)</div>', unsafe_allow_html=True)
        if "new_cases" in filtered.columns:
            box_data = []
            for country in selected_countries_trim[:4]:  # Limit to 4 for readability
                cdf = filtered[filtered["location"] == country]
                if not cdf["new_cases"].dropna().empty:
                    box_data.append(go.Box(y=cdf["new_cases"].dropna(), name=country, marker_color=LINE_PALETTE[selected_countries_trim.index(country) % len(LINE_PALETTE)]))
            if box_data:
                fig_box = go.Figure(data=box_data)
                fig_box.update_layout(
                    template=PLOT_TEMPLATE,
                    title="Daily New Cases Distribution",
                    yaxis_title="Daily New Cases",
                    height=420
                )
                if use_log_scale:
                    fig_box.update_layout(yaxis_type="log")
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("No daily new cases data for box plot.")

        # Daily New Cases Line Chart (existing)
        st.markdown("---")
        st.markdown('<div class="section">Daily New Cases (First 3 Countries)</div>', unsafe_allow_html=True)
        daily_fig = go.Figure()
        for i, country in enumerate(selected_countries_trim[:3]):
            cdf = filtered[filtered["location"] == country]
            if "new_cases" in cdf.columns and not cdf["new_cases"].dropna().empty:
                daily_fig.add_trace(go.Scatter(x=cdf["date"], y=cdf["new_cases"], name=country, mode="lines", line={"width": 2, "color": LINE_PALETTE[i % len(LINE_PALETTE)]}))
        daily_fig.update_layout(
            template=PLOT_TEMPLATE,
            title="Daily New Cases Comparison",
            xaxis_title="Date",
            yaxis_title="Daily New Cases",
            height=420,
            legend={"orientation": "h", "y": -0.2}
        )
        if use_log_scale:
            daily_fig.update_layout(yaxis_type="log")
        st.plotly_chart(daily_fig, use_container_width=True)

# ---------- Vaccination Focus Tab ----------
with tab_vax:
    st.markdown('<div class="section">Vaccination Focus</div>', unsafe_allow_html=True)
    vax_metrics = ["total_vaccinations", "people_vaccinated_per_hundred", "people_fully_vaccinated_per_hundred"]
    if not any(col in df.columns for col in vax_metrics):
        st.info("No vaccination columns in dataset.")
    else:
        latest = safe_latest_by_location(filtered)
        vax_col = next((col for col in vax_metrics if col in latest.columns), None)
        if vax_col:
            vax_label = vax_col.replace("_", " ").title()
            vax_tbl = latest.dropna(subset=[vax_col]).nlargest(15, vax_col)
            if not vax_tbl.empty:
                fig_vax = px.bar(vax_tbl, x="location", y=vax_col, title=f"Top Countries by {vax_label}", color=vax_col, color_continuous_scale="Greens")
                fig_vax.update_layout(template=PLOT_TEMPLATE)
                if use_log_scale:
                    fig_vax.update_layout(yaxis_type="log")
                fig_vax.update_layout(height=420, margin={"t": 40})
                st.plotly_chart(fig_vax, use_container_width=True)
            else:
                st.info("No vaccination data in selected range.")
        
        st.markdown("---")
        sel_country_for_vax = st.selectbox("Country for Vaccination Trend", options=sorted(df["location"].unique()), index=0)
        country_vdf = filtered[filtered["location"] == sel_country_for_vax]
        if not country_vdf.empty and "total_vaccinations" in country_vdf.columns:
            fig_v = px.line(country_vdf, x="date", y="total_vaccinations", title=f"{sel_country_for_vax} — Total Vaccinations")
            fig_v.update_layout(template=PLOT_TEMPLATE)
            if use_log_scale:
                fig_v.update_layout(yaxis_type="log")
            fig_v.update_layout(height=420)
            st.plotly_chart(fig_v, use_container_width=True)
        else:
            st.info("No vaccination data for selected country/range.")

# ---------- Data Explorer Tab ----------
with tab_data:
    st.markdown('<div class="section">Data Explorer</div>', unsafe_allow_html=True)
    st.markdown("Browse filtered dataset, view summary, and download.")

    if st.checkbox("Show raw data (first 500 rows)", value=False):
        st.dataframe(filtered.head(500).style.format(precision=2))

    st.markdown("**Summary Statistics**")
    st.dataframe(filtered.describe().T, use_container_width=True)

    st.markdown("**Columns and Types**")
    col_info = pd.DataFrame({"Column": filtered.columns, "Type": filtered.dtypes.astype(str), "Non-Null Count": filtered.notnull().sum()})
    st.dataframe(col_info)

    csv_buffer = io.StringIO()
    filtered.to_csv(csv_buffer, index=False)
    st.download_button("Download Filtered CSV", data=csv_buffer.getvalue().encode("utf-8"), file_name="covid_filtered.csv", mime="text/csv")

# ---------- News Tab ----------
with tab_news:
    st.markdown('<div class="section">Latest COVID-19 News & Breakouts</div>', unsafe_allow_html=True)
    news_items = fetch_covid_news()
    if news_items:
        for item in news_items:
            st.markdown(f'<div class="news-item"><h4><a href="{item["link"]}" target="_blank">{item["title"]}</a></h4><small>{item["pubDate"]}</small><p>{item["description"][:200]}...</p></div>', unsafe_allow_html=True)
    else:
        st.info("No news available at the moment. Check back later.")