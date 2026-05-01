"""
app.py — NYC Restaurant Health & Rating Dashboard
run: streamlit run app.py
pip install streamlit pandas numpy plotly pydeck scikit-learn joblib
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="NYC Restaurant Intelligence",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.hero-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2.5rem 2rem; border-radius: 16px; margin-bottom: 2rem;
    border: 1px solid rgba(255,255,255,0.08); }
.hero-title { font-family: 'DM Serif Display', serif; font-size: 2.6rem;
    color: #ffffff; margin: 0 0 0.4rem 0; line-height: 1.2; }
.hero-sub { color: rgba(255,255,255,0.6); font-size: 1rem; font-weight: 300; margin: 0; }
.hero-badge { display: inline-block; background: rgba(230,57,70,0.2); color: #e63946;
    border: 1px solid rgba(230,57,70,0.4); border-radius: 20px; padding: 0.2rem 0.8rem;
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.05em; margin-bottom: 0.8rem; }
.metric-card { background: #ffffff; border: 1px solid #e8eaed; border-radius: 12px;
    padding: 1.2rem 1.4rem; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
.metric-label { font-size: 0.78rem; color: #6b7280; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.3rem; }
.metric-value { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #111827; line-height: 1; }
.metric-delta { font-size: 0.82rem; color: #6b7280; margin-top: 0.2rem; }
.metric-delta.bad  { color: #e63946; }
.metric-delta.good { color: #2a9d8f; }
.section-title { font-family: 'DM Serif Display', serif; font-size: 1.6rem;
    color: #ffffff; margin: 0 0 1rem 0; padding-bottom: 0.5rem;
    border-bottom: 2px solid #e63946; display: inline-block; }
[data-testid="stSidebar"] { background: #1a1a2e; }
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label {
    color: rgba(255,255,255,0.6) !important; font-size: 0.82rem !important;
    text-transform: uppercase; letter-spacing: 0.05em; }
.predict-card { border-radius: 14px; padding: 1.8rem; text-align: center; margin-top: 1rem; }
.predict-card.fail { background: linear-gradient(135deg, #fff5f5, #ffe0e0); border: 2px solid #e63946; }
.predict-card.pass { background: linear-gradient(135deg, #f0fdf4, #dcfce7); border: 2px solid #2a9d8f; }
.predict-result { font-family: 'DM Serif Display', serif; font-size: 2.2rem; margin: 0.5rem 0; }
.predict-prob { font-size: 1rem; color: #4b5563; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} .stDeployButton {display: none;}
h3 { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

BASE = os.path.dirname(os.path.abspath(__file__))
FIG  = os.path.join(BASE, "figures")

@st.cache_data
def load_data():
    # df   = pd.read_csv(os.path.join(BASE,'raw','restaurant_clean.csv'),
    #                     low_memory=False, parse_dates=['inspection_date'])
    # yelp = pd.read_csv(os.path.join(BASE,'raw','restaurant_yelp_subset.csv'),
    #                     low_memory=False, parse_dates=['inspection_date'])
    df   = pd.read_csv(os.path.join(BASE,'data','processed','restaurant_clean.csv'),
                    low_memory=False, parse_dates=['inspection_date'])
    yelp = pd.read_csv(os.path.join(BASE,'data','processed','restaurant_yelp_subset.csv'),
                    low_memory=False, parse_dates=['inspection_date'])
    return df, yelp

@st.cache_resource
def load_model():
    import joblib
    path = os.path.join(BASE,'models','best_model.pkl')
    return joblib.load(path) if os.path.exists(path) else None

def fp(name):
    return os.path.join(FIG, name)

def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

df, yelp = load_data()
model    = load_model()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 1.5rem 0;'>
        <div style='font-family:DM Serif Display,serif;font-size:1.3rem;color:white;'>
            🍽️ NYC Restaurant<br>Intelligence
        </div>
        <div style='color:rgba(255,255,255,0.4);font-size:0.75rem;margin-top:0.3rem;'>
             STAT GR5243/GU4243 · Team 15
        </div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("Navigate",
        ["📊 Overview","🗺️ Map Explorer","📈 EDA Insights","🤖 Prediction","ℹ️ About"],
        label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<div style='font-size:0.75rem;color:rgba(255,255,255,0.4);'>Filters</div>",
                unsafe_allow_html=True)
    selected_boro    = st.multiselect("Borough", sorted(df['boro'].unique()), default=sorted(df['boro'].unique()))
    selected_year    = st.multiselect("Year", sorted(df['inspection_year'].unique()), default=sorted(df['inspection_year'].unique()))
    top_cuisines     = df['cuisine_grouped'].value_counts().head(10).index.tolist()
    selected_cuisine = st.multiselect("Cuisine (top 10)", top_cuisines, default=[])

mask = df['boro'].isin(selected_boro) & df['inspection_year'].isin(selected_year)
if selected_cuisine:
    mask &= df['cuisine_grouped'].isin(selected_cuisine)
fdf = df[mask].copy()

# ══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">NYC DOHMH · YELP · OPEN-METEO · CENSUS</div>
        <div class="hero-title">Predicting Restaurant Inspection</div>
        <div class="hero-title">Failures in New York City</div>
        <div class="hero-title" style="color:#e63946; font-size:1.8rem;">A Multi-Source Machine Learning Approach</div>
        <p class="hero-sub">Predicting inspection failures and customer ratings across 22,000+ NYC restaurants
        using regulatory, environmental, and socioeconomic data.</p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    for col,label,value,delta,dc in [
        (c1,"Restaurants",  f"{fdf['camis'].nunique():,}",   "unique establishments",""),
        (c2,"Inspections",  f"{len(fdf):,}",                "2023 – 2024",""),
        (c3,"Failure Rate", f"{fdf['failed'].mean():.1%}",  "score ≥ 28","bad"),
        (c4,"Avg Score",    f"{fdf['score'].mean():.1f}",   "lower is better","good"),
        (c5,"Yelp Coverage",f"{fdf['has_yelp'].mean():.1%}","restaurants matched",""),
    ]:
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-delta {dc}">{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        section("Failure Rate by Borough")
        bs = fdf.groupby('boro').agg(fail_rate=('failed','mean')).reset_index().sort_values('fail_rate',ascending=True)
        fig = px.bar(bs, x='fail_rate', y='boro', orientation='h',
            color='fail_rate', color_continuous_scale=['#2a9d8f','#e9c46a','#e63946'],
            labels={'fail_rate':'Failure Rate','boro':''},
            text=bs['fail_rate'].apply(lambda x: f'{x:.1%}'))
        fig.update_traces(textposition='outside')
        fig.update_layout(coloraxis_showscale=False, plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(l=10,r=30,t=10,b=10), height=300,
            xaxis=dict(tickformat='.0%',showgrid=True,gridcolor='#f0f0f0',tickfont=dict(color='#333')),
            yaxis=dict(showgrid=False,tickfont=dict(color='#333')))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("Monthly Inspections Trend")
        fdf2 = fdf.copy()
        fdf2['year_month'] = fdf2['inspection_date'].dt.to_period('M').astype(str)
        trend = fdf2.groupby(['year_month','boro'])['failed'].count().reset_index()
        trend.columns = ['year_month','boro','count']
        fig2 = px.line(trend, x='year_month', y='count', color='boro',
            color_discrete_sequence=['#e63946','#457b9d','#2a9d8f','#e9c46a','#6d6875'],
            labels={'year_month':'','count':'Inspections','boro':'Borough'})
        fig2.update_layout(plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(l=10,r=10,t=10,b=10), height=300,
            xaxis=dict(showgrid=False,tickangle=45,tickfont=dict(color='#333')),
            yaxis=dict(showgrid=True,gridcolor='#f0f0f0',tickfont=dict(color='#333')),
            legend=dict(orientation='h',yanchor='bottom',y=1.02))
        st.plotly_chart(fig2, use_container_width=True)

    section("Failure Rate by Cuisine Type")
    cs = fdf.groupby('cuisine_grouped').agg(fail_rate=('failed','mean'),count=('failed','count')).reset_index()
    cs = cs[cs['count']>=50].sort_values('fail_rate',ascending=False)
    fig3 = px.bar(cs, x='cuisine_grouped', y='fail_rate',
        color='fail_rate', color_continuous_scale=['#2a9d8f','#e9c46a','#e63946'],
        labels={'cuisine_grouped':'','fail_rate':'Failure Rate'},
        text=cs['fail_rate'].apply(lambda x: f'{x:.1%}'))
    fig3.update_traces(textposition='outside')
    fig3.update_layout(coloraxis_showscale=False, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=10,r=10,t=10,b=10), height=340,
        xaxis=dict(showgrid=False,tickfont=dict(color='#333')),
        yaxis=dict(tickformat='.0%',showgrid=True,gridcolor='#f0f0f0',tickfont=dict(color='#333')))
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — MAP EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == "🗺️ Map Explorer":
    section("Restaurant Map Explorer")
    st.markdown('<p style="color:rgba(255,255,255,0.7);">Showing restaurants with valid coordinates. Color = inspection outcome.</p>',
                unsafe_allow_html=True)

    map_df = fdf[fdf['latitude'].notna() & fdf['longitude'].notna()].drop_duplicates(subset='camis').copy()

    col1,col2 = st.columns([2,1])
    with col1:
        color_by = st.selectbox("Color by",["Inspection Result (failed)","Score Bucket","Borough","Cuisine"])
    with col2:
        st.metric("Restaurants on map", f"{len(map_df):,}")

    if color_by == "Inspection Result (failed)":
        map_df['color'] = map_df['failed'].map({1:[230,57,70,180],0:[42,157,143,180]})
        legend = "🔴 Failed  🟢 Passed"
    elif color_by == "Score Bucket":
        bc = {'A':[42,157,143,180],'B':[233,196,106,180],'C':[230,57,70,180],'NA':[150,150,150,180]}
        map_df['color'] = map_df['score_bucket'].map(bc)
        legend = "🟢 A  🟡 B  🔴 C  ⚪ NA"
    elif color_by == "Borough":
        bc2 = {'MANHATTAN':[69,123,157,180],'BROOKLYN':[230,57,70,180],
               'QUEENS':[42,157,143,180],'BRONX':[233,196,106,180],'STATEN ISLAND':[109,104,117,180]}
        map_df['color'] = map_df['boro'].map(bc2)
        legend = "🔵 Manhattan  🔴 Brooklyn  🟢 Queens  🟡 Bronx  ⚫ Staten Island"
    else:
        map_df['color'] = [[100,100,200,180]]*len(map_df)
        legend = ""

    st.caption(legend)
    try:
        import pydeck as pdk
        map_df['color'] = map_df['color'].apply(lambda x: x if isinstance(x,list) else [150,150,150,180])
        layer = pdk.Layer('ScatterplotLayer',
            data=map_df[['latitude','longitude','color','dba','boro','cuisine','score','failed']],
            get_position='[longitude, latitude]', get_color='color', get_radius=80, pickable=True)
        r = pdk.Deck(layers=[layer],
            initial_view_state=pdk.ViewState(latitude=40.730,longitude=-73.935,zoom=10,pitch=0),
            tooltip={"text":"{dba}\n{boro} · {cuisine}\nScore: {score} | Failed: {failed}"},
            map_style='mapbox://styles/mapbox/light-v10')
        st.pydeck_chart(r)
    except ImportError:
        st.map(map_df[['latitude','longitude']].rename(columns={'latitude':'lat','longitude':'lon'}))

    st.markdown(f"**{len(map_df):,}** restaurants displayed · Failure rate: **{map_df['failed'].mean():.1%}**")

# ══════════════════════════════════════════════════════════════
# PAGE 3 — EDA INSIGHTS
# ══════════════════════════════════════════════════════════════
elif page == "📈 EDA Insights":
    section("Exploratory Data Analysis")

    tab1,tab2,tab3,tab4,tab5 = st.tabs(
        ["📊 Univariate","🔗 Bivariate","🌍 Interactions","🔵 PCA","⭐ Yelp & Neighborhood"])

    with tab1:
        st.markdown("### Core Distributions")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(fp("fig01a_outcome_score.png")):
                st.image(fp("fig01a_outcome_score.png"), caption="Outcome Counts & Score Distribution", use_container_width=True)
            if os.path.exists(fp("fig01c_grade_timing.png")):
                st.image(fp("fig01c_grade_timing.png"), caption="Grade Distribution & Monthly Volume", use_container_width=True)
        with col2:
            if os.path.exists(fp("fig01b_violations.png")):
                st.image(fp("fig01b_violations.png"), caption="Violation Count Distributions", use_container_width=True)
            if os.path.exists(fp("fig02a_borough_cuisine.png")):
                st.image(fp("fig02a_borough_cuisine.png"), caption="Borough & Cuisine Distributions", use_container_width=True)
        if os.path.exists(fp("fig02b_yelp_rating_price.png")):
            st.image(fp("fig02b_yelp_rating_price.png"), caption="Yelp Rating & Price Tier", use_container_width=True)

    with tab2:
        st.markdown("### Bivariate Relationships")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(fp("fig03a_score_violation.png")):
                st.image(fp("fig03a_score_violation.png"), caption="Score by Outcome & Violation Failure Rate", use_container_width=True)
            if os.path.exists(fp("fig03c_trend_firstrepeat.png")):
                st.image(fp("fig03c_trend_firstrepeat.png"), caption="Score Trend & First vs Repeat", use_container_width=True)
        with col2:
            if os.path.exists(fp("fig03b_critical_history.png")):
                st.image(fp("fig03b_critical_history.png"), caption="Critical Violations & Historical Scores", use_container_width=True)
            if os.path.exists(fp("fig04b_monthly_temp_failure.png")):
                st.image(fp("fig04b_monthly_temp_failure.png"), caption="Monthly & Temperature Failure Rates", use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(fp("fig04a_yelp_vs_outcome.png")):
                st.image(fp("fig04a_yelp_vs_outcome.png"), caption="Yelp Rating vs Inspection Outcome", use_container_width=True)
        with col2:
            if os.path.exists(fp("fig08_target_correlations.png")):
                st.image(fp("fig08_target_correlations.png"), caption="Feature Correlations with Failure Target", use_container_width=True)

    with tab3:
        st.markdown("### Interaction Effects")
        if os.path.exists(fp("fig06a_income_complaint_heatmap.png")):
            st.image(fp("fig06a_income_complaint_heatmap.png"), caption="Income x 311 Complaint Failure Rate", use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(fp("fig06b_cuisine_borough_heatmap.png")):
                st.image(fp("fig06b_cuisine_borough_heatmap.png"), caption="Cuisine x Borough Failure Rate", use_container_width=True)
        with col2:
            if os.path.exists(fp("fig06c_borough_month_heatmap.png")):
                st.image(fp("fig06c_borough_month_heatmap.png"), caption="Borough x Month Failure Rate", use_container_width=True)
        if os.path.exists(fp("fig07_correlation_heatmap.png")):
            st.image(fp("fig07_correlation_heatmap.png"), caption="Numeric Predictor Correlation Heatmap", use_container_width=True)

    with tab4:
        st.markdown("### PCA — Dimensionality Reduction")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(fp("fig09a_pca_variance.png")):
                st.image(fp("fig09a_pca_variance.png"), caption="Scree Plot & Cumulative Variance", use_container_width=True)
            if os.path.exists(fp("fig10a_pca_loadings.png")):
                st.image(fp("fig10a_pca_loadings.png"), caption="PC1 & PC2 Feature Loadings", use_container_width=True)
        with col2:
            if os.path.exists(fp("fig09b_pca_projection.png")):
                st.image(fp("fig09b_pca_projection.png"), caption="PCA 2D Projection by Outcome", use_container_width=True)
            if os.path.exists(fp("fig10b_pc1_borough.png")):
                st.image(fp("fig10b_pc1_borough.png"), caption="PC1 Distribution by Borough", use_container_width=True)
        st.markdown("### KMeans Clustering")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(fp("fig11_kmeans_selection.png")):
                st.image(fp("fig11_kmeans_selection.png"), caption="KMeans Elbow & Silhouette", use_container_width=True)
            if os.path.exists(fp("fig12a_cluster_overview.png")):
                st.image(fp("fig12a_cluster_overview.png"), caption="Cluster Size & Failure Rate", use_container_width=True)
        with col2:
            if os.path.exists(fp("fig12b_cluster_profiles_pca.png")):
                st.image(fp("fig12b_cluster_profiles_pca.png"), caption="Cluster Profiles & PCA Projection", use_container_width=True)
            if os.path.exists(fp("fig13_cluster_composition.png")):
                st.image(fp("fig13_cluster_composition.png"), caption="Cluster Composition by Borough & Outcome", use_container_width=True)

    with tab5:
        st.markdown("### Yelp & Neighborhood")
        yelp_fdf = yelp[yelp['boro'].isin(selected_boro)].copy()
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("**Yelp Rating Distribution**")
            fig5 = px.histogram(yelp_fdf[yelp_fdf['yelp_rating'].notna()],
                x='yelp_rating', nbins=9, color_discrete_sequence=['#457b9d'],
                labels={'yelp_rating':'Yelp Rating'})
            fig5.update_layout(plot_bgcolor='white',paper_bgcolor='white',
                margin=dict(l=10,r=10,t=10,b=10),height=300,bargap=0.1,
                xaxis=dict(tickfont=dict(color='#333')),yaxis=dict(tickfont=dict(color='#333')))
            st.plotly_chart(fig5, use_container_width=True)
        with col2:
            st.markdown("**Yelp Rating vs Inspection Score**")
            sy = yelp_fdf[yelp_fdf['score'].notna() & yelp_fdf['yelp_rating'].notna()]
            fig6 = px.scatter(sy, x='yelp_rating', y='score', color='failed',
                color_discrete_map={0:'#2a9d8f',1:'#e63946'}, trendline='ols', opacity=0.6,
                labels={'yelp_rating':'Yelp Rating','score':'Inspection Score','failed':'Failed'})
            fig6.update_layout(plot_bgcolor='white',paper_bgcolor='white',
                margin=dict(l=10,r=10,t=10,b=10),height=300,
                xaxis=dict(tickfont=dict(color='#333')),yaxis=dict(tickfont=dict(color='#333')))
            st.plotly_chart(fig6, use_container_width=True)

        col1,col2 = st.columns(2)
        with col1:
            st.markdown("**Income vs Failure Rate by Borough**")
            inf = fdf.groupby('boro').agg(fail_rate=('failed','mean'),
                income=('median_household_income','mean'),count=('failed','count')).reset_index()
            fig7 = px.scatter(inf, x='income', y='fail_rate', size='count', color='boro', text='boro',
                color_discrete_sequence=['#e63946','#457b9d','#2a9d8f','#e9c46a','#6d6875'],
                labels={'income':'Median Household Income ($)','fail_rate':'Failure Rate'})
            fig7.update_traces(textposition='top center')
            fig7.update_layout(plot_bgcolor='white',paper_bgcolor='white',
                margin=dict(l=10,r=10,t=10,b=10),height=300,showlegend=False,
                yaxis=dict(tickformat='.0%',showgrid=True,gridcolor='#f0f0f0'))
            st.plotly_chart(fig7, use_container_width=True)
        with col2:
            st.markdown("**311 Food Complaints vs Failure Rate**")
            cf = fdf.groupby('boro').agg(fail_rate=('failed','mean'),
                complaints=('food_complaints_total','mean')).reset_index()
            fig8 = px.scatter(cf, x='complaints', y='fail_rate', color='boro', text='boro',
                color_discrete_sequence=['#e63946','#457b9d','#2a9d8f','#e9c46a','#6d6875'],
                labels={'complaints':'Avg Daily 311 Food Complaints','fail_rate':'Failure Rate'})
            fig8.update_traces(textposition='top center')
            fig8.update_layout(plot_bgcolor='white',paper_bgcolor='white',
                margin=dict(l=10,r=10,t=10,b=10),height=300,showlegend=False,
                yaxis=dict(tickformat='.0%',showgrid=True,gridcolor='#f0f0f0'))
            st.plotly_chart(fig8, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 4 — PREDICTION
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Prediction":
    st.markdown('<h2 style="color:#e63946;font-size:1.8rem;font-weight:700;margin-bottom:0.5rem;">Inspection Failure Predictor</h2>',
                unsafe_allow_html=True)
    st.markdown('<p style="color:rgba(255,255,255,0.7);margin-bottom:1.5rem;">Enter restaurant characteristics to predict the probability of failing a health inspection.</p>',
                unsafe_allow_html=True)
    if model is None:
        st.warning("⚠️ Model file not found. Place `best_model.pkl` in the `models/` folder. Showing demo mode.")

    col1,col2,col3 = st.columns(3)
    with col1:
        st.markdown("**Restaurant Info**")
        boro_input             = st.selectbox("Borough", sorted(df['boro'].unique()))
        cuisine_input          = st.selectbox("Cuisine", sorted(df['cuisine_grouped'].unique()))
        inspection_count_input = st.slider("Number of past inspections", 1, 20, 3)
    with col2:
        st.markdown("**Inspection History**")
        prev_score_input      = st.slider("Previous inspection score", -1, 50, 12, help="-1 = first inspection")
        prev_failed_input     = st.selectbox("Previous inspection result", [-1,0,1],
            format_func=lambda x: {-1:"N/A (first)",0:"Passed",1:"Failed"}[x])
        violation_count_input = st.slider("Number of violations this visit", 0, 10, 2)
        critical_count_input  = st.slider("Number of critical violations", 0, 5, 1)
    with col3:
        st.markdown("**Context**")
        temp_input       = st.slider("Temperature on inspection day (°C)", -10.0, 35.0, 18.0)
        precip_input     = st.slider("Precipitation (mm)", 0.0, 50.0, 0.0)
        month_input      = st.selectbox("Month", list(range(1,13)),
            format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
        is_weekend_input = st.checkbox("Weekend inspection")

    st.markdown("---")

    if st.button("🔍 Predict Inspection Outcome", type="primary", use_container_width=True):
        if model is not None:
            try:
                boro_df = df[df['boro'] == boro_input]
                input_data = pd.DataFrame([{
                    'prev_score':              prev_score_input,
                    'prev_failed':             prev_failed_input,
                    'inspection_count':        inspection_count_input,
                    'poor_history_flag':       int(prev_failed_input == 1 or prev_score_input >= 28),
                    'is_first_inspection':     int(inspection_count_input == 1),
                    'has_history':             int(prev_score_input != -1),
                    'food_complaints_total':   df['food_complaints_total'].mean(),
                    'rodent_complaints':       df['rodent_complaints'].mean(),
                    'food_safety_complaints':  df['food_safety_complaints'].mean(),
                    'complaint_intensity':     (df['food_complaints_total'].mean()
                                                + df['rodent_complaints'].mean()
                                                + df['food_safety_complaints'].mean()),
                    'complaint_density':       (df['food_complaints_total'].mean()
                                                / boro_df['total_population'].mean() * 10000),
                    'high_complaint_flag':     0,
                    'temp_mean':               temp_input,
                    'precipitation_sum':       precip_input,
                    'rain_sum':                precip_input,
                    'snowfall_sum':            0.0,
                    'wind_speed_mean':         df['wind_speed_mean'].mean(),
                    'cloud_cover_mean':        df['cloud_cover_mean'].mean(),
                    'month_sin':               np.sin(2 * np.pi * month_input / 12),
                    'month_cos':               np.cos(2 * np.pi * month_input / 12),
                    'dow_sin':                 0.0,
                    'dow_cos':                 1.0,
                    'summer_flag':             int(month_input in [6,7,8]),
                    'is_weekend':              int(is_weekend_input),
                    'median_household_income': boro_df['median_household_income'].mean(),
                    'total_population':        boro_df['total_population'].mean(),
                    'white_pct':               boro_df['white_pct'].mean(),
                    'has_yelp':                0,
                    'has_location':            0,
                    'log_yelp_reviews':        0.0,
                    'yelp_price':              np.nan,
                    'boro':                    boro_input,
                    'cuisine_grouped':         cuisine_input,
                    'yelp_category_primary':   'unknown',
                }])
                prob = model.predict_proba(input_data)[0][1]
                pred = int(prob >= 0.47)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                prob, pred = 0.18, 0
        else:
            prob = min(0.95, 0.05 + critical_count_input*0.15 + violation_count_input*0.05
                       + (0.1 if prev_failed_input == 1 else 0))
            pred = int(prob >= 0.47)

        _,c2,_ = st.columns([1,2,1])
        with c2:
            cc   = "fail" if pred else "pass"
            icon = "⚠️" if pred else "✅"
            rtxt = "LIKELY TO FAIL" if pred else "LIKELY TO PASS"
            col  = "#e63946" if pred else "#2a9d8f"
            st.markdown(f"""<div class="predict-card {cc}">
                <div style="font-size:3rem;">{icon}</div>
                <div class="predict-result" style="color:{col};">{rtxt}</div>
                <div class="predict-prob">Failure probability: <strong>{prob:.1%}</strong></div>
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=prob*100,
                domain={'x':[0,1],'y':[0,1]},
                title={'text':"Failure Probability (%)"},
                gauge={'axis':{'range':[0,100]},
                       'bar':{'color':"#e63946" if pred else "#2a9d8f"},
                       'steps':[{'range':[0,30],'color':'#dcfce7'},
                                 {'range':[30,60],'color':'#fef9c3'},
                                 {'range':[60,100],'color':'#fee2e2'}],
                       'threshold':{'line':{'color':"black",'width':2},'thickness':0.8,'value':47}}))
            fig_g.update_layout(height=280,margin=dict(l=20,r=20,t=40,b=20),paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_g, use_container_width=True)
            st.caption("Model: Logistic Regression · Threshold: 0.47 · AUC: 0.630 · Recall: 63.7% · F1: 0.354")

# ══════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    section("About This Project")
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("""
### Predicting Restaurant Inspection Failures in New York City
**A Multi-Source Machine Learning Approach**

This dashboard presents an end-to-end machine learning pipeline built for
Columbia University GU4243/GR5243 Applied Data Science, Project 4, Team 15.

#### Research Questions
- Can we predict whether a NYC restaurant will **fail** its next health inspection?
- What factors most strongly influence **Yelp ratings** among inspected restaurants?

#### Data Sources
| Source | Method | Records |
|--------|--------|---------|
| NYC DOHMH Inspections | Socrata API | 142,591 raw |
| Restaurant Coordinates | Socrata API | 4,487 geocoded |
| Yelp Business Data | Yelp Fusion API | ~3,000 matched |
| Weather (Open-Meteo) | REST API | 731 daily |
| NYC 311 Complaints | Socrata API | Food-related |
| U.S. Census ACS 2022 | Census API | 5 boroughs |
| Violation Codes | Web Scraping | Reference table |
""")
    with col2:
        st.markdown("""
#### Pipeline Overview
```
1. Data Acquisition    → 6 sources, 4 methods
2. Data Cleaning       → 41,556 clean records
3. EDA + Clustering    → Borough behavior patterns
4. Feature Engineering → 52 features, leakage-safe
5. Modeling            → Logistic / RF / Gradient Boosting
6. Model Selection     → Best model by AUC + Recall
7. Web App             → This dashboard
```


""")

    st.markdown("---")
    section("Key Findings")
    c1,c2,c3,c4 = st.columns(4)
    for col,label,val,cap in [
        (c1,"Final Model", "Logistic Regression","best AUC + recall balance"),
        (c2,"ROC-AUC",     "0.630",              "threshold = 0.47"),
        (c3,"Recall",      "63.7%",              "failing restaurants caught"),
        (c4,"F1 Score",    "0.354",              "precision–recall balance"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-delta">{cap}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if os.path.exists(fp("model_comparison_table.png")):
        st.image(fp("model_comparison_table.png"), use_container_width=True)

    col1,col2 = st.columns(2)
    with col1:
        if os.path.exists(fp("model_performance_comparison.png")):
            st.image(fp("model_performance_comparison.png"), caption="Model Performance Comparison", use_container_width=True)
    with col2:
        if os.path.exists(fp("model_roc_comparison.png")):
            st.image(fp("model_roc_comparison.png"), caption="ROC Curve Comparison", use_container_width=True)

    col1,col2 = st.columns(2)
    with col1:
        if os.path.exists(fp("final_model_confusion_matrix.png")):
            st.image(fp("final_model_confusion_matrix.png"), caption="Confusion Matrix (threshold=0.47)", use_container_width=True)
    with col2:
        if os.path.exists(fp("final_model_feature_importance.png")):
            st.image(fp("final_model_feature_importance.png"), caption="Top Predictors — Logistic Regression", use_container_width=True)

    col1,col2 = st.columns(2)
    with col1:
        st.markdown("""
#### Model Comparison Summary
| Model | AUC | Recall | F1 |
|-------|-----|--------|----|
| **Logistic Regression** ✅ | **0.630** | **63.7%** | **0.354** |
| Random Forest | 0.614 | 31.0% | 0.289 |
| Gradient Boosting | 0.616 | 0.02% | 0.000 |

Logistic Regression was selected as the final model because it achieves the best
balance of recall, F1-score, and AUC, while remaining fully interpretable —
critical in a public health context where regulators need to trust model decisions.
""")
    with col2:
        st.markdown("""
#### Substantive Findings
- **Inspection failure rate**: 17.9% across all NYC boroughs (2023–2024)
- **Historical violations** are the strongest predictor of future failures
- **Bronx** has the highest failure rate; **Staten Island** has the lowest
- **Yelp rating** shows weak correlation with inspection outcomes
- **Threshold tuning** 0.50 → 0.47 improved recall from 53.8% to 63.7%
""")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#9ca3af;font-size:0.85rem;'>"
        "Columbia University · GU4243/GR5243 Applied Data Science · Spring 2026 · Team 15"
        # "Columbia University · STAT GR5243/GU4243 Applied Data Science · Fall 2026 · Team 15"
        "</div>", unsafe_allow_html=True)
