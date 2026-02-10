import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Personal Energy & Focus Analyzer",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
 
df = pd.read_csv("personal_energy_focus_data.csv")
df["date"] = pd.to_datetime(df["date"])

# Feature engineering (recreated for deployment)
df["productivity_score"] = (
    df["focus_level"] * 0.5 +
    df["energy_level"] * 0.3 +
    df["mood_score"] * 0.2
).round(2)

df["week"] = df["date"].dt.isocalendar().week

# ---------------- SIDEBAR ----------------
st.sidebar.title("Filters")

workday_filter = st.sidebar.multiselect(
    "Day Type",
    options=df["workday"].unique(),
    default=df["workday"].unique()
)

filtered_df = df[df["workday"].isin(workday_filter)]

# ---------------- HEADER ----------------
st.title(" Personal Energy & Focus Analyzer")
st.caption("Exploratory Data Analysis using Python & Streamlit")

# ---------------- KPIs ----------------
c1, c2, c3 = st.columns(3)

c1.metric("Average Focus", round(filtered_df["focus_level"].mean(), 2))
c2.metric("Average Energy", round(filtered_df["energy_level"].mean(), 2))
c3.metric("Productivity Score", round(filtered_df["productivity_score"].mean(), 2))

st.divider()

# ---------------- WORKDAY COMPARISON ----------------
st.subheader(" Workday vs Weekend Analysis")

summary = filtered_df.groupby("workday")[
    ["focus_level", "energy_level", "productivity_score"]
].mean().round(2)

st.dataframe(summary)

st.divider()

# ---------------- SLEEP VS FOCUS ----------------
st.subheader(" Sleep vs Focus")

fig1 = plt.figure()
plt.scatter(filtered_df["sleep_hours"], filtered_df["focus_level"])
plt.xlabel("Sleep Hours")
plt.ylabel("Focus Level")
plt.title("Impact of Sleep on Focus")
st.pyplot(fig1)

# ---------------- SCREEN TIME VS ENERGY ----------------
st.subheader(" Screen Time vs Energy")

fig2 = plt.figure()
plt.scatter(filtered_df["screen_time_hours"], filtered_df["energy_level"])
plt.xlabel("Screen Time (hours)")
plt.ylabel("Energy Level")
plt.title("Impact of Screen Time on Energy")
st.pyplot(fig2)

st.divider()

# ---------------- WEEKLY TREND ----------------
st.subheader(" Weekly Productivity Trend")

weekly = filtered_df.groupby("week")["productivity_score"].mean()

fig3 = plt.figure()
plt.plot(weekly)
plt.xlabel("Week")
plt.ylabel("Average Productivity")
plt.title("Productivity Trend Over Time")
st.pyplot(fig3)

st.divider()

# ---------------- CORRELATION ----------------
st.subheader(" Correlation Matrix")

corr = filtered_df[
    [
        "sleep_hours",
        "screen_time_hours",
        "mood_score",
        "energy_level",
        "focus_level",
        "productivity_score"
    ]
].corr()

st.dataframe(corr.round(2))

