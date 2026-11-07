"""Hertfordshire Road Collision Dashboard (2021–2025).

Run from the project root:
    .venv/bin/streamlit run app/dashboard.py
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
CLEAN, GEO = ROOT / "data/clean", ROOT / "data/geo"

RED, ORANGE, GREY, BLUE, DARK = "#dc2626", "#f97316", "#9ca3af", "#2563eb", "#111827"
SEV_COLORS = {"Fatal": "#7f1d1d", "Serious": ORANGE, "Slight": GREY}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
HERTS_CENTER = {"lat": 51.81, "lon": -0.24}
# ONS Census 2021 usual residents, rounded (see notebook 02).
POP_2021 = {"Broxbourne": 99_700, "Dacorum": 155_100, "East Hertfordshire": 151_200, "Hertsmere": 107_200,
            "North Hertfordshire": 133_600, "St Albans": 148_900, "Stevenage": 89_700, "Three Rivers": 94_300,
            "Watford": 102_300, "Welwyn Hatfield": 119_900}

st.set_page_config(page_title="Herts Road Collisions", page_icon="🚦", layout="wide")


# ---------------------------------------------------------------- data
@st.cache_data
def load():
    c = pd.read_parquet(CLEAN / "collisions.parquet")
    k = pd.read_parquet(CLEAN / "casualties.parquet")
    c["severity"] = pd.Categorical(c["severity"].astype(str), ["Fatal", "Serious", "Slight"], ordered=True)
    c["time_band"] = c["time_band"].astype(str)
    k = k.merge(c[["collision_index", "year", "district"]], on="collision_index")
    hotspots = pd.read_csv(CLEAN / "hotspots_dbscan.csv")
    cells = gpd.read_file(CLEAN / "gi_star_cells.geojson")
    districts = gpd.read_file(GEO / "herts_districts.geojson")[["LAD24NM", "geometry"]]
    return c, k, hotspots, cells, districts


@st.cache_data
def load_model():
    return json.loads((ROOT / "models/ksi_logit.json").read_text())


c_all, k_all, hotspots, cells, districts = load()
MODEL = load_model()


# ---------------------------------------------------------------- sidebar filters
st.sidebar.title("🚦 Filters")
yr = st.sidebar.slider("Years", 2021, 2025, (2021, 2025))
dist = st.sidebar.multiselect("District", sorted(c_all.district.unique()), placeholder="All districts")
sev = st.sidebar.multiselect("Severity", ["Fatal", "Serious", "Slight"], default=["Fatal", "Serious", "Slight"])
users = st.sidebar.multiselect(
    "Involving", ["Pedestrian hurt", "Cyclist hurt", "Motorcyclist hurt", "HGV", "E-scooter", "Young driver (17–24)",
                  "Older driver (70+)"], placeholder="Anyone")
light = st.sidebar.radio("Light", ["All", "Daylight", "Dark"], horizontal=True)
road_cls = st.sidebar.multiselect("Road class", ["Motorway", "A(M)", "A", "B", "C", "Unclassified"], placeholder="All roads")

USER_FILTERS = {
    "Pedestrian hurt": lambda d: d.pedestrian_casualties > 0, "Cyclist hurt": lambda d: d.cyclist_casualties > 0,
    "Motorcyclist hurt": lambda d: d.motorcyclist_casualties > 0, "HGV": lambda d: d.involves_hgv,
    "E-scooter": lambda d: d.involves_escooter, "Young driver (17–24)": lambda d: d.involves_young_driver,
    "Older driver (70+)": lambda d: d.involves_older_driver,
}

c = c_all[c_all.year.between(*yr) & c_all.severity.isin(sev)]
if dist:
    c = c[c.district.isin(dist)]
for u in users:
    c = c[USER_FILTERS[u](c)]
if light != "All":
    c = c[c.is_dark == (light == "Dark")]
if road_cls:
    c = c[c.first_road_class_label.isin(road_cls)]
k = k_all[k_all.collision_index.isin(c.collision_index)]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(c):,}** of {len(c_all):,} collisions")
st.sidebar.caption("Source: DfT STATS19 road safety open data, Open Government Licence. "
                   "Boundaries: ONS. Map: © OpenStreetMap contributors.")


# ---------------------------------------------------------------- helpers
def pct(x):
    return f"{x:.1%}" if pd.notna(x) else "–"


def empty_guard():
    if c.empty:
        st.warning("No collisions match these filters. Try widening them in the sidebar.")
        st.stop()


def style(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), legend_title_text="",
                      font=dict(family="Inter, system-ui, sans-serif"))
    return fig


def ksi_rate(df, by, min_n=20):
    t = df.groupby(by, observed=True).agg(collisions=("is_ksi", "size"), ksi=("is_ksi", "sum")).reset_index()
    t["ksi_rate"] = t.ksi / t.collisions
    return t[t.collisions >= min_n]


# ---------------------------------------------------------------- header + KPIs
st.title("Hertfordshire Road Collisions")
st.caption(f"Police-reported injury collisions, {yr[0]}–{yr[1]} · "
           f"{'All districts' if not dist else ', '.join(dist)}")
empty_guard()

n_years = yr[1] - yr[0] + 1
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Collisions", f"{len(c):,}", help="Police-reported collisions where someone was injured")
k2.metric("KSI", f"{c.is_ksi.sum():,}", help="Collisions where someone was Killed or Seriously Injured")
k3.metric("KSI rate", pct(c.is_ksi.mean()), help="Share of collisions that were KSI")
k4.metric("Fatal", f"{(c.collision_severity == 1).sum():,}", help="Collisions where someone died")
vu = k.casualty_group.isin(["Pedestrian", "Cyclist", "Motorcyclist"])
k5.metric("Vulnerable KSI", pct(k[k.is_ksi & vu].shape[0] / max(k.is_ksi.sum(), 1)),
          help="Pedestrians, cyclists and motorcyclists as a share of killed or seriously injured casualties")

tab_over, tab_when, tab_where, tab_who, tab_hot, tab_pred = st.tabs(
    ["📈 Overview", "🕒 When", "📍 Where", "🧍 Who", "🔥 Hotspots", "🔮 What-if"])


# ---------------------------------------------------------------- Overview
with tab_over:
    a, b = st.columns([3, 2])
    yearly = c.groupby(["year", "severity"], observed=True).size().reset_index(name="collisions")
    fig = px.bar(yearly, x="year", y="collisions", color="severity", color_discrete_map=SEV_COLORS,
                 category_orders={"severity": ["Slight", "Serious", "Fatal"]}, title="Collisions per year by severity")
    fig.update_xaxes(dtick=1)
    a.plotly_chart(style(fig), width="stretch")

    monthly = c.set_index("date").resample("MS").agg(collisions=("is_ksi", "size"), ksi=("is_ksi", "sum")).reset_index()
    monthly["12-month average"] = monthly.collisions.rolling(12, min_periods=6).mean()
    fig = go.Figure([
        go.Scatter(x=monthly.date, y=monthly.collisions, name="Monthly", line=dict(color=GREY, width=1)),
        go.Scatter(x=monthly.date, y=monthly["12-month average"], name="12-month average", line=dict(color=BLUE, width=3)),
        go.Scatter(x=monthly.date, y=monthly.ksi, name="KSI", line=dict(color=RED, width=1.5)),
    ])
    fig.update_layout(title="Monthly trend")
    b.plotly_chart(style(fig), width="stretch")

    st.subheader("Key facts for this selection")
    hr = ksi_rate(c, "time_band", 10)
    dark = ksi_rate(c, "is_dark", 10).set_index("is_dark").ksi_rate
    roads = c[~c.road_name.isin(["Unclassified", "Unknown"])].groupby("road_name").is_ksi.sum().nlargest(3)
    facts = []
    if len(hr):
        worst = hr.sort_values("ksi_rate").iloc[-1]
        facts.append(f"**Most dangerous time band:** {worst.time_band}, where {worst.ksi_rate:.0%} of collisions are KSI "
                     f"(average {c.is_ksi.mean():.0%}).")
    if {True, False} <= set(dark.index):
        facts.append(f"**Darkness:** KSI rate {dark[True]:.0%} in the dark vs {dark[False]:.0%} in daylight.")
    busiest = c.groupby(["weekday", "hour"]).size().idxmax()
    facts.append(f"**Busiest hour:** {busiest[0]} at {busiest[1]:02d}:00.")
    if len(roads):
        facts.append("**Roads with most KSI collisions:** " + ", ".join(f"{r} ({n})" for r, n in roads.items()) + ".")
    for f in facts:
        st.markdown(f"- {f}")


# ---------------------------------------------------------------- When
with tab_when:
    grid = c.pivot_table(index="weekday", columns="hour", values="collision_index", aggfunc="size") \
            .reindex(index=DAYS, columns=range(24)).fillna(0)
    fig = px.imshow(grid, color_continuous_scale="Reds", aspect="auto", labels=dict(x="Hour of day", y="", color="Collisions"),
                    title="Collisions by hour and day of week")
    st.plotly_chart(style(fig, 330), width="stretch")

    a, b = st.columns(2)
    hourly = ksi_rate(c, "hour", 1)
    fig = go.Figure([go.Bar(x=hourly.hour, y=hourly.collisions, name="Collisions", marker_color=GREY),
                     go.Scatter(x=hourly.hour, y=hourly.ksi_rate, name="KSI rate", yaxis="y2",
                                line=dict(color=RED, width=2.5), mode="lines+markers")])
    fig.update_layout(title="Volume vs danger by hour", xaxis_title="Hour",
                      yaxis=dict(title="Collisions"), yaxis2=dict(title="KSI rate", overlaying="y", side="right",
                                                                   tickformat=".0%", showgrid=False),
                      legend=dict(orientation="h", y=1.1))
    a.plotly_chart(style(fig), width="stretch")

    m = ksi_rate(c, "month_name", 1)
    m["month_name"] = pd.Categorical(m.month_name, MONTHS, ordered=True)
    fig = px.bar(m.sort_values("month_name"), x="month_name", y="collisions", color="ksi_rate",
                 color_continuous_scale="Reds", title="Collisions by month (colour = KSI rate)",
                 labels={"month_name": "", "ksi_rate": "KSI rate"})
    fig.update_coloraxes(colorbar_tickformat=".0%")
    b.plotly_chart(style(fig), width="stretch")


# ---------------------------------------------------------------- Where
with tab_where:
    a, b = st.columns([3, 2])
    pts = c.dropna(subset=["latitude"]).sort_values("collision_severity", ascending=False)
    sample_note = ""
    if len(pts) > 6000:
        pts = pd.concat([pts[pts.is_ksi], pts[~pts.is_ksi].sample(6000 - pts.is_ksi.sum(), random_state=1)]) \
              if pts.is_ksi.sum() < 6000 else pts.sample(6000, random_state=1)
        sample_note = " (all KSI shown; slight collisions sampled for speed)"
    pts = pts.sort_values("collision_severity", ascending=False)
    fig = px.scatter_map(pts, lat="latitude", lon="longitude", color="severity", color_discrete_map=SEV_COLORS,
                         hover_name="road_name",
                         hover_data={"district": True, "date": "|%d %b %Y", "time": True, "latitude": False,
                                     "longitude": False, "severity": True},
                         zoom=8.8, center=HERTS_CENTER, map_style="open-street-map", opacity=.75,
                         title="Collision locations" + sample_note)
    fig.update_traces(marker=dict(size=6))
    a.plotly_chart(style(fig, 560), width="stretch", config={"scrollZoom": False})

    d = c.groupby("district").agg(collisions=("is_ksi", "size"), ksi=("is_ksi", "sum")).reset_index()
    d["KSI rate"] = d.ksi / d.collisions
    d["KSI per 10k residents / yr"] = d.ksi / d.district.map(POP_2021) * 10_000 / n_years
    d = d.sort_values("KSI per 10k residents / yr", ascending=True)
    fig = px.bar(d, y="district", x="KSI per 10k residents / yr", orientation="h", color="KSI rate",
                 color_continuous_scale="Reds", title="KSI per 10,000 residents per year",
                 hover_data={"collisions": True, "ksi": True, "KSI rate": ":.1%"}, labels={"district": ""})
    fig.update_coloraxes(colorbar_tickformat=".0%")
    b.plotly_chart(style(fig, 560), width="stretch")
    st.caption("Per-resident rates favour districts with less through-traffic: Hertsmere and St Albans carry the M1 and M25.")

    a, b = st.columns(2)
    sl = ksi_rate(c.dropna(subset=["speed_limit"]).assign(speed=lambda x: x.speed_limit.astype(int).astype(str) + " mph"), "speed")
    sl = sl.assign(order=sl.speed.str.extract(r"(\d+)").astype(int)).sort_values("order")
    fig = px.bar(sl, x="speed", y="ksi_rate", text=sl.collisions.map(lambda n: f"n={n:,}"),
                 title="KSI rate by speed limit", labels={"speed": "", "ksi_rate": "KSI rate"}, color_discrete_sequence=[BLUE])
    fig.add_hline(c.is_ksi.mean(), line_dash="dash", line_color=RED, annotation_text="average")
    fig.update_yaxes(tickformat=".0%")
    a.plotly_chart(style(fig), width="stretch")

    roads = ksi_rate(c[~c.road_name.isin(["Unclassified", "Unknown"])], "road_name", 15).nlargest(12, "ksi")
    fig = px.bar(roads.sort_values("ksi"), y="road_name", x=["collisions", "ksi"], orientation="h", barmode="overlay",
                 title="Roads with the most KSI collisions", labels={"road_name": "", "value": "Collisions", "variable": ""},
                 color_discrete_map={"ksi": RED, "collisions": GREY})
    fig.for_each_trace(lambda t: t.update(name={"ksi": "KSI", "collisions": "All"}[t.name]))
    b.plotly_chart(style(fig), width="stretch")


# ---------------------------------------------------------------- Who
with tab_who:
    if k.empty:
        st.info("No casualties in this selection.")
    else:
        a, b = st.columns(2)
        g = k.groupby("casualty_group").agg(casualties=("is_ksi", "size"), ksi=("is_ksi", "sum")).reset_index()
        g["KSI rate"] = g.ksi / g.casualties
        g = g.sort_values("ksi")
        fig = px.bar(g, y="casualty_group", x="ksi", orientation="h", color_discrete_sequence=[RED],
                     text=(g.ksi / g.ksi.sum()).map("{:.0%}".format),
                     title="Killed or seriously injured casualties by road user (share of all KSI)",
                     hover_data={"casualties": True, "KSI rate": ":.0%"}, labels={"casualty_group": "", "ksi": "KSI casualties"})
        a.plotly_chart(style(fig), width="stretch")

        g2 = g[g.casualties >= 20].sort_values("KSI rate")
        fig = px.bar(g2, y="casualty_group", x="KSI rate", orientation="h", text=g2["KSI rate"].map("{:.0%}".format),
                     title="Chance a casualty is killed or seriously injured", labels={"casualty_group": ""},
                     color_discrete_sequence=[BLUE])
        fig.add_vline(k.is_ksi.mean(), line_dash="dash", line_color=RED)
        fig.update_xaxes(tickformat=".0%")
        b.plotly_chart(style(fig), width="stretch")

        ages = k.dropna(subset=["age_of_casualty"])
        ages = ages[ages.casualty_group.isin(["Car occupant", "Pedestrian", "Cyclist", "Motorcyclist"])]
        fig = px.histogram(ages, x="age_of_casualty", color="casualty_group", nbins=45, barmode="overlay", opacity=.6,
                           title="Age of casualties", labels={"age_of_casualty": "Age", "casualty_group": ""})
        st.plotly_chart(style(fig, 340), width="stretch")


# ---------------------------------------------------------------- Hotspots
with tab_hot:
    st.markdown("Repeat-collision sites found with **DBSCAN** (8+ collisions within 100 m) and areas of statistically "
                "significant harm found with **Getis-Ord Gi\\*** (500 m cells). These use all 2021–25 data. "
                "Only the **District** filter applies here.")
    hs = hotspots if not dist else hotspots[hotspots.district.isin(dist)]
    hs = hs.copy()
    hs["site"] = hs["site"].fillna(hs["location"])
    n_show = st.slider("Number of sites to show", 5, min(50, max(len(hs), 5)), min(20, max(len(hs), 5)))
    top = hs.head(n_show)

    show_gi = st.toggle("Show Gi* hot spot cells", value=True)
    fig = go.Figure()
    if show_gi:
        hot = cells[cells.hotspot.str.startswith("Hot")]
        if dist:
            hot = hot[hot.district.isin(dist)]
        hot = hot.assign(cls=hot.hotspot.map({"Hot spot (FDR)": 2, "Hot spot 99%": 1, "Hot spot 95%": 1}))
        if len(hot):
            fig.add_trace(go.Choroplethmap(
                geojson=json.loads(hot[["cell_id", "geometry"]].to_json()), locations=hot.cell_id,
                featureidkey="properties.cell_id", z=hot.cls, colorscale=[[0, "#f87171"], [1, "#7f1d1d"]],
                zmin=1, zmax=2, showscale=False, marker_opacity=.45, marker_line_width=0,
                customdata=np.c_[hot.hotspot, hot.collisions.astype(int), hot.ksi.astype(int)],
                hovertemplate="%{customdata[0]}<br>%{customdata[1]} collisions · %{customdata[2]} KSI<extra></extra>",
                name="Gi* hot spot"))
    dsel = districts if not dist else districts[districts.LAD24NM.isin(dist)]
    for _, r in dsel.iterrows():
        geoms = r.geometry.geoms if r.geometry.geom_type == "MultiPolygon" else [r.geometry]
        for gm in geoms:
            x, y = gm.exterior.xy
            fig.add_trace(go.Scattermap(lon=list(x), lat=list(y), mode="lines", line=dict(color="#374151", width=1.2),
                                        hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scattermap(
        lat=top.latitude, lon=top.longitude, mode="markers+text",
        marker=dict(size=np.clip(top.score / 3, 12, 34), color=DARK, opacity=.85),
        text=top["rank"].astype(str), textfont=dict(color="white", size=11),
        customdata=np.c_[top.site, top.district, top.collisions, top.ksi, top.ksi_rate.map("{:.0%}".format),
                         top.years_with_collision],
        hovertemplate="<b>#%{text} %{customdata[0]}</b><br>%{customdata[1]}<br>%{customdata[2]} collisions · "
                      "%{customdata[3]} KSI (%{customdata[4]})<br>Active %{customdata[5]}/5 years<extra></extra>",
        name="Hotspot sites"))
    center = HERTS_CENTER if not dist else {"lat": top.latitude.mean(), "lon": top.longitude.mean()} if len(top) else HERTS_CENTER
    fig.update_layout(map=dict(style="open-street-map", center=center, zoom=9.2 if not dist else 11), showlegend=False)
    st.plotly_chart(style(fig, 620), width="stretch", config={"scrollZoom": False})

    table = top[["rank", "site", "district", "collisions", "fatal", "serious", "ksi", "ksi_rate", "pct_vulnerable",
                 "pct_dark", "peak_time", "speed_limit", "years_with_collision", "early_2021_23", "late_2024_25", "gi_hotspot"]].copy()
    table[["ksi_rate", "pct_vulnerable", "pct_dark"]] = table[["ksi_rate", "pct_vulnerable", "pct_dark"]] * 100
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "rank": "#", "site": "Site", "district": "District", "collisions": "Collisions", "fatal": "Fatal",
            "serious": "Serious", "ksi": "KSI",
            "ksi_rate": st.column_config.ProgressColumn("KSI rate", format="%.0f%%", min_value=0, max_value=100),
            "pct_vulnerable": st.column_config.NumberColumn("Vulnerable hurt", format="%.0f%%"),
            "pct_dark": st.column_config.NumberColumn("In dark", format="%.0f%%"),
            "peak_time": "Peak time", "speed_limit": st.column_config.NumberColumn("Speed limit", format="%d mph"),
            "years_with_collision": "Years active", "early_2021_23": "2021–23", "late_2024_25": "2024–25",
            "gi_hotspot": "Gi* result",
        })
    st.download_button("Download hotspot table (CSV)", top.to_csv(index=False), "herts_hotspots.csv", "text/csv")


# ---------------------------------------------------------------- What-if (severity model)
FEATURE_LABELS = {
    "time_band": "Time of day", "day_type": "Day", "district": "District", "speed_limit": "Speed limit",
    "road_class": "Road type", "area": "Area", "light": "Light", "weather": "Weather", "surface": "Road surface",
    "junction": "Junction", "vehicles": "Vehicles involved", "pedestrian": "Pedestrian hit",
    "cyclist": "Cyclist involved", "motorcycle": "Motorcycle involved", "hgv": "HGV / lorry involved",
    "bus": "Bus or minibus involved", "escooter": "E-scooter involved", "young_driver": "Young driver (17–24)",
    "older_driver": "Older driver (70+)",
}
ORDERS = {
    "time_band": ["Night (00–05)", "Morning peak (06–09)", "Daytime (10–15)", "Evening peak (16–18)",
                  "Evening (19–21)", "Late (22–23)"],
    "speed_limit": ["20 mph", "30 mph", "40 mph", "50 mph", "60 mph", "70 mph"],
    "road_class": ["Motorway", "A", "B", "Unclassified"],
    "light": ["Daylight", "Dark, street lights on", "Dark, no street lights"],
    "vehicles": ["1 (single vehicle)", "2", "3 or more"],
}
YES_NO = ["pedestrian", "cyclist", "motorcycle", "hgv", "bus", "escooter", "young_driver", "older_driver"]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def options(f):
    opts = MODEL["features"][f]["options"]
    order = ORDERS.get(f)
    return [o for o in order if o in opts] + [o for o in opts if o not in order] if order else sorted(opts)


with tab_pred:
    st.markdown("**If a collision happens under these conditions, how likely is it that someone is killed or "
                "seriously injured?** Pick the conditions below; the estimate updates instantly.")

    choice = {}
    st.markdown("##### Where and when")
    cols = st.columns(4)
    for i, f in enumerate(["time_band", "day_type", "district", "speed_limit", "road_class", "area", "junction", "vehicles"]):
        opts = options(f)
        choice[f] = cols[i % 4].selectbox(FEATURE_LABELS[f], opts, index=opts.index(MODEL["features"][f]["default"]),
                                          key=f"wi_{f}")
    st.markdown("##### Conditions")
    cols = st.columns(3)
    for i, f in enumerate(["light", "weather", "surface"]):
        opts = options(f)
        choice[f] = cols[i].selectbox(FEATURE_LABELS[f], opts, index=opts.index(MODEL["features"][f]["default"]),
                                      key=f"wi_{f}")
    st.markdown("##### Who was involved")
    TOGGLE_LABELS = {"pedestrian": "Pedestrian", "cyclist": "Cyclist", "motorcycle": "Motorcycle",
                     "hgv": "HGV / lorry", "bus": "Bus", "escooter": "E-scooter",
                     "young_driver": "Driver 17–24", "older_driver": "Driver 70+"}
    cols = st.columns(4)
    for i, f in enumerate(YES_NO):
        choice[f] = "Yes" if cols[i % 4].toggle(TOGGLE_LABELS[f], key=f"wi_{f}", help=FEATURE_LABELS[f]) else "No"

    feats = MODEL["features"]
    contrib = {f: feats[f]["coef"][v] - feats[f]["mean_coef"] for f, v in choice.items()}
    z_avg = MODEL["intercept"] + sum(feats[f]["mean_coef"] for f in feats)
    prob = sigmoid(z_avg + sum(contrib.values()))
    avg = MODEL["base_rate"]

    st.markdown("---")
    ratio = prob / avg
    word = "higher than" if ratio > 1.05 else "lower than" if ratio < 0.95 else "about the same as"
    colour = RED if ratio > 1.05 else BLUE if ratio < 0.95 else "#4b5563"
    a, b = st.columns([2, 3], gap="large")
    with a:
        st.markdown(
            f"<div style='font-size:.95rem;color:#4b5563'>Chance someone is killed or seriously injured</div>"
            f"<div style='font-size:3.2rem;font-weight:700;line-height:1.1;color:{colour}'>{prob:.0%}</div>"
            f"<div style='color:{colour};margin-bottom:.4rem'><b>{ratio:.1f}×</b> the Hertfordshire average "
            f"({avg:.0%}): {word} average</div>", unsafe_allow_html=True)
        gauge = go.Figure(go.Indicator(
            mode="gauge", value=prob * 100,
            gauge=dict(axis=dict(range=[0, 100], ticksuffix="%", tickvals=[0, 25, 50, 75, 100]),
                       bar=dict(color=DARK, thickness=.3),
                       steps=[dict(range=[0, 15], color="#dcfce7"), dict(range=[15, 30], color="#fef9c3"),
                              dict(range=[30, 50], color="#fed7aa"), dict(range=[50, 100], color="#fecaca")],
                       threshold=dict(line=dict(color=RED, width=3), value=avg * 100))))
        gauge.update_layout(height=190, margin=dict(l=40, r=40, t=10, b=10))
        st.plotly_chart(gauge, width="stretch", config={"displayModeBar": False})
        st.caption("Red line on the dial = Hertfordshire average.")

    with b:
        st.markdown("**What's driving this estimate**  \n"
                    "<span style='color:#4b5563;font-size:.9rem'>How each choice changes the odds compared with an "
                    "average crash</span>", unsafe_allow_html=True)
        cdf = pd.DataFrame([{"factor": f"{FEATURE_LABELS[f]}" if f in YES_NO else f"{FEATURE_LABELS[f]}: {choice[f]}",
                             "mult": np.exp(v), "flag_off": f in YES_NO and choice[f] == "No"}
                            for f, v in contrib.items()])
        cdf = cdf[~cdf.flag_off & ((cdf.mult > 1.05) | (cdf.mult < 0.95))].sort_values("mult")
        if cdf.empty:
            st.info("These are all fairly typical conditions, so the risk is close to average.")
        else:
            cdf["effect"] = np.where(cdf.mult > 1, "Raises risk", "Lowers risk")
            cdf["text"] = cdf.mult.map(lambda m: f"×{m:.1f}" if m >= 1 else f"×{m:.2f}")
            fig = go.Figure(go.Bar(
                y=cdf.factor, x=cdf.mult - 1, base=1, orientation="h", text=cdf.text, textposition="outside",
                marker_color=np.where(cdf.mult > 1, RED, BLUE), cliponaxis=False,
                hovertemplate="%{y}<br>%{text} odds<extra></extra>"))
            lo, hi = min(0.4, cdf.mult.min() * 0.7), max(2.5, cdf.mult.max() * 1.6)
            ticks = [t for t in [0.25, 0.5, 1, 2, 4, 8] if lo <= t <= hi]
            fig.update_xaxes(type="log", range=[np.log10(lo), np.log10(hi)], tickvals=ticks,
                             ticktext=[{0.25: "×¼", 0.5: "×½"}.get(t, f"×{t:g}") for t in ticks], title="")
            fig.add_vline(1, line_color="black", line_width=1)
            fig.update_layout(showlegend=False, height=max(220, 36 * len(cdf) + 70),
                              margin=dict(l=10, r=30, t=10, b=10), font=dict(family="Inter, system-ui, sans-serif"))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.caption("Red = raises risk, blue = lowers it. Only choices that change the odds by more than 5% are shown.")

    m = MODEL["test_metrics_2025"]
    with st.expander("About this model, and how far to trust it"):
        st.markdown(f"""
- **What it is:** a logistic regression trained on {MODEL['n_train']:,} police-reported injury collisions in
  Hertfordshire (DfT STATS19, 2021–2025). Every option above has one weight; the dashboard just adds them up.
- **What it predicts:** how *severe* a crash is likely to be **if it happens**. It does **not** predict the chance
  of a crash happening.
- **How good it is:** tested on 2025 data it never saw while learning, it scores ROC AUC **{m['roc_auc']:.2f}**
  (0.5 = guessing, 1 = perfect). The 20% of crashes it rates riskiest contained **{m['top20_capture']:.0%}** of the
  serious ones. A more complex gradient-boosting model scored {m['gbm_roc_auc']:.2f}, no better.
- **Limits:** the open data has no impact speed, seatbelt use or vehicle safety ratings, which matter a lot. So read
  this as *relative* risk. The factors are associations, not proof of cause. For example, "rural" partly
  stands in for higher real speeds.
- **Full method:** see `notebooks/04_model.ipynb` in the GitHub repo.
""")
