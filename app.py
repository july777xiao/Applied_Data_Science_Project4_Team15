"""
app.py — NYC Restaurant Health & Rating Dashboard
运行方式: streamlit run app.py
依赖: pip install streamlit pandas numpy plotly pydeck scikit-learn
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import pickle

# ─────────────────────────────────────────────────────────────
# 页面配置
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Restaurant Intelligence",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# 全局样式
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* 顶部横幅 */
.hero-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid rgba(255,255,255,0.08);
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #ffffff;
    margin: 0 0 0.4rem 0;
    line-height: 1.2;
}
.hero-sub {
    color: rgba(255,255,255,0.6);
    font-size: 1rem;
    font-weight: 300;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(230,57,70,0.2);
    color: #e63946;
    border: 1px solid rgba(230,57,70,0.4);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 0.8rem;
}

/* 指标卡片 */
.metric-card {
    background: #ffffff;
    border: 1px solid #e8eaed;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.metric-label {
    font-size: 0.78rem;
    color: #6b7280;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #111827;
    line-height: 1;
}
.metric-delta {
    font-size: 0.82rem;
    color: #6b7280;
    margin-top: 0.2rem;
}
.metric-delta.bad  { color: #e63946; }
.metric-delta.good { color: #2a9d8f; }

/* 区块标题 */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #111827;
    margin: 0 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e63946;
    display: inline-block;
}

/* 侧边栏 */
[data-testid="stSidebar"] {
    background: #1a1a2e;
}
[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.85) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label {
    color: rgba(255,255,255,0.6) !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* 预测结果卡 */
.predict-card {
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
    margin-top: 1rem;
}
.predict-card.fail {
    background: linear-gradient(135deg, #fff5f5, #ffe0e0);
    border: 2px solid #e63946;
}
.predict-card.pass {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 2px solid #2a9d8f;
}
.predict-result {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    margin: 0.5rem 0;
}
.predict-prob {
    font-size: 1rem;
    color: #4b5563;
}

/* 隐藏Streamlit默认元素 */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    clean_path = os.path.join(base, 'raw', 'restaurant_clean.csv')
    yelp_path  = os.path.join(base, 'raw', 'restaurant_yelp_subset.csv')
    df   = pd.read_csv(clean_path,  low_memory=False, parse_dates=['inspection_date'])
    yelp = pd.read_csv(yelp_path,   low_memory=False, parse_dates=['inspection_date'])
    return df, yelp

@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base, 'models', 'best_model.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

df, yelp = load_data()
model    = load_model()

# ─────────────────────────────────────────────────────────────
# 侧边栏导航
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 1.5rem 0;'>
        <div style='font-family: DM Serif Display, serif; font-size: 1.3rem; color: white;'>
            🍽️ NYC Restaurant<br>Intelligence
        </div>
        <div style='color: rgba(255,255,255,0.4); font-size: 0.75rem; margin-top: 0.3rem;'>
            Project 4 · Columbia GR5243
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["📊 Overview", "🗺️ Map Explorer", "📈 EDA Insights", "🤖 Prediction", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("<div style='font-size:0.75rem; color:rgba(255,255,255,0.4);'>Filters</div>",
                unsafe_allow_html=True)

    selected_boro = st.multiselect(
        "Borough",
        options=sorted(df['boro'].unique()),
        default=sorted(df['boro'].unique())
    )
    selected_year = st.multiselect(
        "Year",
        options=sorted(df['inspection_year'].unique()),
        default=sorted(df['inspection_year'].unique())
    )
    top_cuisines = df['cuisine_grouped'].value_counts().head(10).index.tolist()
    selected_cuisine = st.multiselect(
        "Cuisine (top 10)",
        options=top_cuisines,
        default=[]
    )

# 过滤数据
mask = (
    df['boro'].isin(selected_boro) &
    df['inspection_year'].isin(selected_year)
)
if selected_cuisine:
    mask &= df['cuisine_grouped'].isin(selected_cuisine)
fdf = df[mask].copy()

# ─────────────────────────────────────────────────────────────
# PAGE 1: OVERVIEW
# ─────────────────────────────────────────────────────────────
if page == "📊 Overview":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">NYC DOHMH · YELP · OPEN-METEO · CENSUS</div>
        <div class="hero-title">Health, Hygiene & Hype</div>
        <div class="hero-title" style="color: #e63946;">NYC Restaurant Intelligence</div>
        <p class="hero-sub">
            Predicting inspection failures and customer ratings across 22,000+ NYC restaurants
            using regulatory, environmental, and socioeconomic data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI 指标
    c1, c2, c3, c4, c5 = st.columns(5)

    total       = len(fdf['camis'].unique())
    fail_rate   = fdf['failed'].mean()
    total_insp  = len(fdf)
    avg_score   = fdf['score'].mean()
    yelp_pct    = fdf['has_yelp'].mean()

    for col, label, value, delta, delta_class in [
        (c1, "Restaurants",      f"{total:,}",          "unique establishments",   ""),
        (c2, "Inspections",      f"{total_insp:,}",     "2023 – 2024",             ""),
        (c3, "Failure Rate",     f"{fail_rate:.1%}",    "score ≥ 28",              "bad"),
        (c4, "Avg Score",        f"{avg_score:.1f}",    "lower is better",         "good"),
        (c5, "Yelp Coverage",    f"{yelp_pct:.1%}",     "restaurants matched",     ""),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-delta {delta_class}">{delta}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 行政区失败率 + 检查数量趋势
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Failure Rate by Borough</div>', unsafe_allow_html=True)
        boro_stats = fdf.groupby('boro').agg(
            fail_rate=('failed', 'mean'),
            count=('failed', 'count')
        ).reset_index().sort_values('fail_rate', ascending=True)

        fig = px.bar(
            boro_stats, x='fail_rate', y='boro', orientation='h',
            color='fail_rate',
            color_continuous_scale=['#2a9d8f', '#e9c46a', '#e63946'],
            labels={'fail_rate': 'Failure Rate', 'boro': ''},
            text=boro_stats['fail_rate'].apply(lambda x: f'{x:.1%}')
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            coloraxis_showscale=False,
            plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(l=10, r=30, t=10, b=10),
            height=300,
            xaxis=dict(tickformat='.0%', showgrid=True, gridcolor='#f0f0f0'),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Monthly Inspections Trend</div>', unsafe_allow_html=True)
        fdf['year_month'] = fdf['inspection_date'].dt.to_period('M').astype(str)
        trend = fdf.groupby(['year_month', 'boro'])['failed'].agg(['count', 'mean']).reset_index()
        trend.columns = ['year_month', 'boro', 'count', 'fail_rate']

        fig2 = px.line(
            trend, x='year_month', y='count', color='boro',
            color_discrete_sequence=['#e63946', '#457b9d', '#2a9d8f', '#e9c46a', '#6d6875'],
            labels={'year_month': '', 'count': 'Inspections', 'boro': 'Borough'}
        )
        fig2.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
            xaxis=dict(showgrid=False, tickangle=45),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # 菜系失败率
    st.markdown('<div class="section-title">Failure Rate by Cuisine Type</div>', unsafe_allow_html=True)
    cuisine_stats = fdf.groupby('cuisine_grouped').agg(
        fail_rate=('failed', 'mean'),
        count=('failed', 'count')
    ).reset_index()
    cuisine_stats = cuisine_stats[cuisine_stats['count'] >= 50].sort_values('fail_rate', ascending=False)

    fig3 = px.bar(
        cuisine_stats, x='cuisine_grouped', y='fail_rate',
        color='fail_rate',
        color_continuous_scale=['#2a9d8f', '#e9c46a', '#e63946'],
        labels={'cuisine_grouped': '', 'fail_rate': 'Failure Rate'},
        text=cuisine_stats['fail_rate'].apply(lambda x: f'{x:.1%}')
    )
    fig3.update_traces(textposition='outside')
    fig3.update_layout(
        coloraxis_showscale=False,
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
        xaxis=dict(showgrid=False),
        yaxis=dict(tickformat='.0%', showgrid=True, gridcolor='#f0f0f0'),
    )
    st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PAGE 2: MAP EXPLORER
# ─────────────────────────────────────────────────────────────
elif page == "🗺️ Map Explorer":
    st.markdown('<div class="section-title">Restaurant Map Explorer</div>', unsafe_allow_html=True)
    st.markdown("Showing restaurants with valid coordinates. Color = inspection outcome.")

    map_df = fdf[fdf['latitude'].notna() & fdf['longitude'].notna()].copy()
    map_df = map_df.drop_duplicates(subset='camis')

    col1, col2 = st.columns([2, 1])
    with col1:
        color_by = st.selectbox(
            "Color by",
            ["Inspection Result (failed)", "Score Bucket", "Borough", "Cuisine"]
        )
    with col2:
        st.metric("Restaurants on map", f"{len(map_df):,}")

    if color_by == "Inspection Result (failed)":
        map_df['color'] = map_df['failed'].map({1: [230, 57, 70, 180], 0: [42, 157, 143, 180]})
        legend = "🔴 Failed  🟢 Passed"
    elif color_by == "Score Bucket":
        bucket_colors = {'A': [42, 157, 143, 180], 'B': [233, 196, 106, 180],
                         'C': [230, 57, 70, 180], 'NA': [150, 150, 150, 180]}
        map_df['color'] = map_df['score_bucket'].map(bucket_colors)
        legend = "🟢 A  🟡 B  🔴 C  ⚪ NA"
    elif color_by == "Borough":
        boro_colors = {
            'MANHATTAN': [69, 123, 157, 180], 'BROOKLYN': [230, 57, 70, 180],
            'QUEENS': [42, 157, 143, 180], 'BRONX': [233, 196, 106, 180],
            'STATEN ISLAND': [109, 104, 117, 180]
        }
        map_df['color'] = map_df['boro'].map(boro_colors)
        legend = "🔵 Manhattan  🔴 Brooklyn  🟢 Queens  🟡 Bronx  ⚫ Staten Island"
    else:
        map_df['color'] = [[100, 100, 200, 180]] * len(map_df)
        legend = ""

    st.caption(legend)

    try:
        import pydeck as pdk
        map_df['color'] = map_df['color'].apply(lambda x: x if isinstance(x, list) else [150,150,150,180])
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=map_df[['latitude', 'longitude', 'color', 'dba', 'boro', 'cuisine', 'score', 'failed']],
            get_position='[longitude, latitude]',
            get_color='color',
            get_radius=80,
            pickable=True,
        )
        view_state = pdk.ViewState(latitude=40.730, longitude=-73.935, zoom=10, pitch=0)
        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "{dba}\n{boro} · {cuisine}\nScore: {score} | Failed: {failed}"},
            map_style='mapbox://styles/mapbox/light-v10',
        )
        st.pydeck_chart(r)
    except ImportError:
        st.map(map_df[['latitude', 'longitude']].rename(columns={'latitude': 'lat', 'longitude': 'lon'}))

    st.markdown("---")
    st.markdown(f"**{len(map_df):,}** restaurants displayed · "
                f"Failure rate in view: **{map_df['failed'].mean():.1%}**")

# ─────────────────────────────────────────────────────────────
# PAGE 3: EDA INSIGHTS
# ─────────────────────────────────────────────────────────────
elif page == "📈 EDA Insights":
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🌡️ Weather & Scores", "📅 Temporal Patterns", "⭐ Yelp Analysis", "🏘️ Neighborhood"]
    )

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Temperature vs Inspection Score")
            sample = fdf[fdf['score'].notna() & fdf['temp_mean'].notna()].sample(min(3000, len(fdf)), random_state=42)
            fig = px.scatter(
                sample, x='temp_mean', y='score',
                color='failed',
                color_discrete_map={0: '#2a9d8f', 1: '#e63946'},
                labels={'temp_mean': 'Mean Temperature (°C)', 'score': 'Inspection Score',
                        'failed': 'Failed'},
                opacity=0.5,
            )
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                              margin=dict(l=10, r=10, t=10, b=10), height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Score Distribution by Weather")
            fdf['precip_cat'] = pd.cut(fdf['precipitation_sum'],
                                        bins=[-0.1, 0, 5, 20, 200],
                                        labels=['No Rain', 'Light', 'Moderate', 'Heavy'])
            weather_score = fdf.groupby('precip_cat')['score'].mean().reset_index()
            fig2 = px.bar(
                weather_score, x='precip_cat', y='score',
                color='score',
                color_continuous_scale=['#2a9d8f', '#e9c46a', '#e63946'],
                labels={'precip_cat': 'Precipitation', 'score': 'Avg Score'}
            )
            fig2.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                               coloraxis_showscale=False,
                               margin=dict(l=10, r=10, t=10, b=10), height=350)
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Failure Rate by Day of Week")
            dow_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
            dow_stats = fdf.groupby('inspection_dow')['failed'].mean().reset_index()
            dow_stats['day'] = dow_stats['inspection_dow'].map(dow_map)
            fig3 = px.bar(
                dow_stats, x='day', y='failed',
                color='failed',
                color_continuous_scale=['#2a9d8f', '#e63946'],
                labels={'day': '', 'failed': 'Failure Rate'}
            )
            fig3.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                               coloraxis_showscale=False,
                               margin=dict(l=10, r=10, t=10, b=10), height=320,
                               yaxis=dict(tickformat='.0%'))
            st.plotly_chart(fig3, use_container_width=True)

        with col2:
            st.subheader("Failure Rate by Month")
            month_map = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                         7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
            month_stats = fdf.groupby('inspection_month')['failed'].mean().reset_index()
            month_stats['month_name'] = month_stats['inspection_month'].map(month_map)
            fig4 = px.line(
                month_stats, x='month_name', y='failed',
                markers=True,
                labels={'month_name': '', 'failed': 'Failure Rate'},
                color_discrete_sequence=['#e63946']
            )
            fig4.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                               margin=dict(l=10, r=10, t=10, b=10), height=320,
                               yaxis=dict(tickformat='.0%', showgrid=True, gridcolor='#f0f0f0'),
                               xaxis=dict(showgrid=False))
            st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        yelp_fdf = yelp[yelp['boro'].isin(selected_boro)].copy()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Yelp Rating Distribution")
            fig5 = px.histogram(
                yelp_fdf[yelp_fdf['yelp_rating'].notna()],
                x='yelp_rating', nbins=9,
                color_discrete_sequence=['#457b9d'],
                labels={'yelp_rating': 'Yelp Rating', 'count': 'Count'}
            )
            fig5.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                               margin=dict(l=10, r=10, t=10, b=10), height=320,
                               bargap=0.1)
            st.plotly_chart(fig5, use_container_width=True)

        with col2:
            st.subheader("Yelp Rating vs Inspection Score")
            sample_yelp = yelp_fdf[yelp_fdf['score'].notna() & yelp_fdf['yelp_rating'].notna()]
            fig6 = px.scatter(
                sample_yelp, x='yelp_rating', y='score',
                color='failed',
                color_discrete_map={0: '#2a9d8f', 1: '#e63946'},
                trendline='ols',
                labels={'yelp_rating': 'Yelp Rating', 'score': 'Inspection Score', 'failed': 'Failed'},
                opacity=0.6,
            )
            fig6.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                               margin=dict(l=10, r=10, t=10, b=10), height=320)
            st.plotly_chart(fig6, use_container_width=True)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Income vs Failure Rate by Borough")
            income_fail = fdf.groupby('boro').agg(
                fail_rate=('failed', 'mean'),
                income=('median_household_income', 'mean'),
                count=('failed', 'count')
            ).reset_index()
            fig7 = px.scatter(
                income_fail, x='income', y='fail_rate',
                size='count', color='boro', text='boro',
                labels={'income': 'Median Household Income ($)',
                        'fail_rate': 'Failure Rate', 'boro': 'Borough'},
                color_discrete_sequence=['#e63946','#457b9d','#2a9d8f','#e9c46a','#6d6875']
            )
            fig7.update_traces(textposition='top center')
            fig7.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                               margin=dict(l=10, r=10, t=10, b=10), height=350,
                               yaxis=dict(tickformat='.0%', showgrid=True, gridcolor='#f0f0f0'),
                               showlegend=False)
            st.plotly_chart(fig7, use_container_width=True)

        with col2:
            st.subheader("311 Food Complaints vs Failure Rate")
            complaint_fail = fdf.groupby('boro').agg(
                fail_rate=('failed', 'mean'),
                complaints=('food_complaints_total', 'mean')
            ).reset_index()
            fig8 = px.scatter(
                complaint_fail, x='complaints', y='fail_rate',
                color='boro', text='boro', size_max=20,
                labels={'complaints': 'Avg Daily 311 Food Complaints',
                        'fail_rate': 'Failure Rate', 'boro': 'Borough'},
                color_discrete_sequence=['#e63946','#457b9d','#2a9d8f','#e9c46a','#6d6875']
            )
            fig8.update_traces(textposition='top center')
            fig8.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                               margin=dict(l=10, r=10, t=10, b=10), height=350,
                               yaxis=dict(tickformat='.0%', showgrid=True, gridcolor='#f0f0f0'),
                               showlegend=False)
            st.plotly_chart(fig8, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PAGE 4: PREDICTION
# ─────────────────────────────────────────────────────────────
elif page == "🤖 Prediction":
    st.markdown('<div class="section-title">Inspection Failure Predictor</div>', unsafe_allow_html=True)
    st.markdown("Enter restaurant characteristics to predict the probability of failing a health inspection.")

    if model is None:
        st.warning("⚠️ Model file not found. Place `best_model.pkl` in the `models/` folder. "
                   "Showing demo interface below.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Restaurant Info**")
        boro_input    = st.selectbox("Borough", sorted(df['boro'].unique()))
        cuisine_input = st.selectbox("Cuisine", sorted(df['cuisine_grouped'].unique()))
        inspection_count_input = st.slider("Number of past inspections", 1, 20, 3)

    with col2:
        st.markdown("**Inspection History**")
        prev_score_input  = st.slider("Previous inspection score", -1, 50, 12,
                                       help="-1 = first inspection")
        prev_failed_input = st.selectbox("Previous inspection result",
                                          [-1, 0, 1],
                                          format_func=lambda x: {-1: "N/A (first)", 0: "Passed", 1: "Failed"}[x])
        violation_count_input = st.slider("Number of violations this visit", 0, 10, 2)
        critical_count_input  = st.slider("Number of critical violations", 0, 5, 1)

    with col3:
        st.markdown("**Context**")
        temp_input   = st.slider("Temperature on inspection day (°C)", -10.0, 35.0, 18.0)
        precip_input = st.slider("Precipitation (mm)", 0.0, 50.0, 0.0)
        month_input  = st.selectbox("Month", list(range(1, 13)),
                                     format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun',
                                                             'Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
        is_weekend_input = st.checkbox("Weekend inspection")

    st.markdown("---")

    if st.button("🔍 Predict Inspection Outcome", type="primary", use_container_width=True):
        if model is not None:
            try:
                input_data = pd.DataFrame([{
                    'violation_count':          violation_count_input,
                    'critical_count':           critical_count_input,
                    'temp_mean':                temp_input,
                    'precipitation_sum':        precip_input,
                    'inspection_month':         month_input,
                    'is_weekend':               int(is_weekend_input),
                    'prev_score':               prev_score_input,
                    'prev_failed':              prev_failed_input,
                    'inspection_count':         inspection_count_input,
                    'score_trend':              0,
                    'has_history':              int(prev_score_input != -1),
                    'is_first_inspection':      int(inspection_count_input == 1),
                    'grade_available':          1,
                    'has_yelp':                 0,
                    'has_location':             0,
                    'food_complaints_total':    df['food_complaints_total'].mean(),
                    'rodent_complaints':        df['rodent_complaints'].mean(),
                    'median_household_income':  df[df['boro'] == boro_input]['median_household_income'].mean(),
                    'white_pct':                df[df['boro'] == boro_input]['white_pct'].mean(),
                }])
                prob = model.predict_proba(input_data)[0][1]
                pred = int(prob >= 0.5)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                prob, pred = 0.18, 0
        else:
            # Demo mode: 简单规则模拟
            prob = min(0.95, 0.05 + critical_count_input * 0.15 + violation_count_input * 0.05
                       + (0.1 if prev_failed_input == 1 else 0))
            pred = int(prob >= 0.5)

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            card_class = "fail" if pred == 1 else "pass"
            icon       = "⚠️" if pred == 1 else "✅"
            result_txt = "LIKELY TO FAIL" if pred == 1 else "LIKELY TO PASS"
            color      = "#e63946" if pred == 1 else "#2a9d8f"

            st.markdown(f"""
            <div class="predict-card {card_class}">
                <div style="font-size: 3rem;">{icon}</div>
                <div class="predict-result" style="color: {color};">{result_txt}</div>
                <div class="predict-prob">
                    Failure probability: <strong>{prob:.1%}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 概率条
            st.markdown("<br>", unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Failure Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#e63946" if pred else "#2a9d8f"},
                    'steps': [
                        {'range': [0, 30],  'color': '#dcfce7'},
                        {'range': [30, 60], 'color': '#fef9c3'},
                        {'range': [60, 100],'color': '#fee2e2'},
                    ],
                    'threshold': {'line': {'color': "black", 'width': 2}, 'thickness': 0.8, 'value': 50}
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20),
                                    paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_gauge, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PAGE 5: ABOUT
# ─────────────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.markdown('<div class="section-title">About This Project</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Health, Hygiene & Hype
        **Predicting Restaurant Inspection Outcomes in NYC**

        This dashboard presents an end-to-end machine learning pipeline built for
        Columbia University GR5243 (Applied Data Science) Project 4.

        #### Research Questions
        - Can we predict whether a NYC restaurant will **fail** its next health inspection?
        - What factors most strongly influence **Yelp ratings** among inspected restaurants?

        #### Data Sources
        | Source | Method | Records |
        |--------|--------|---------|
        | NYC DOHMH Inspections | Socrata API | 142,591 raw |
        | Yelp Business Data | Yelp Fusion API | ~8,300 matched |
        | Weather (Open-Meteo) | REST API | 731 daily |
        | NYC 311 Complaints | Socrata API | Food-related |
        | US Census ACS 2022 | Census API | 5 boroughs |
        | Violation Codes | Web Scraping | Reference table |
        """)

    with col2:
        st.markdown("""
        #### Pipeline Overview
        ```
        1. Data Acquisition    → 6 sources, 4 methods
        2. Data Cleaning       → 41,556 clean records
        3. EDA + Clustering    → Borough behavior patterns
        4. Feature Engineering → 52 features
        5. Modeling            → Logistic / RF / XGBoost
        6. Model Selection     → Best model by AUC + F1
        7. Web App             → This dashboard
        ```

        #### Team Contributions
        | Member | Role |
        |--------|------|
        | [Name A] | Data acquisition & cleaning |
        | [Name B] | EDA & unsupervised learning |
        | [Name C] | Modeling & evaluation |
        | [Name D] | Report & web app |

        #### Key Findings
        - Inspection failure rate: **18.0%** across all boroughs
        - Critical violations are the strongest predictor of failure
        - Bronx has the highest failure rate; Staten Island the lowest
        - Yelp rating shows weak correlation with inspection outcomes
        """)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#9ca3af; font-size:0.85rem;'>"
        "Columbia University · GR5243 Applied Data Science · Spring 2026"
        "</div>",
        unsafe_allow_html=True
    )
