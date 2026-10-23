# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Wildlife Audio Alert Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------
# Simulated detection store (replace later with real model output)
# -------------------------------------------------
if "detections" not in st.session_state:
    st.session_state.detections = []

if "alert_active" not in st.session_state:
    st.session_state.alert_active = False

if "last_alert" not in st.session_state:
    st.session_state.last_alert = None

# -------------------------------------------------
# Helper: create a fake detection (for demo)
# -------------------------------------------------
def generate_demo_detection():
    classes = ["Human", "Gunshot", "Vehicle", "Background"]
    weights = [0.15, 0.10, 0.12, 0.35, 0.15, 0.13]  # Background more common
    label = random.choices(classes, weights=weights)[0]
    confidence = round(random.uniform(0.55, 0.98), 2)
    location = random.choice([
        "Sector A – North Fence",
        "Sector B – River Crossing",
        "Sector C – Waterhole",
        "Sector D – East Gate",
        "Sector E – Ridge Path",
    ])
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label": label,
        "confidence": confidence,
        "location": location,
        "is_threat": label in ["Human", "Gunshot", "Vehicle"],
    }

# -------------------------------------------------
# Sidebar controls
# -------------------------------------------------
with st.sidebar:
    st.title("Dashboard Controls")
    st.markdown("---")

    auto_refresh = st.checkbox("Auto-refresh every 5s", value=True)
    demo_mode = st.checkbox("Demo mode (simulate detections)", value=True)

    if st.button("Clear all detections"):
        st.session_state.detections = []
        st.session_state.alert_active = False
        st.session_state.last_alert = None
        st.rerun()

    st.markdown("---")
    st.info(
        "Threat classes that trigger RED ALERT:\n\n"
        "• Human\n• Gunshot\n• Vehicle"
    )

# -------------------------------------------------
# Header + Alert Banner
# -------------------------------------------------
st.title("Wildlife Audio Surveillance Dashboard")
st.caption("Edge AI • Human / Gunshot / Vehicle detection with OOD rejection")

# Big red alert banner
if st.session_state.alert_active and st.session_state.last_alert:
    alert = st.session_state.last_alert
    st.markdown(
        f"""
        <div style="
            background-color: #8B0000;
            padding: 20px 25px;
            border-radius: 10px;
            color: white;
            margin-bottom: 20px;
            border: 3px solid #FF0000;
            box-shadow: 0 0 15px rgba(255,0,0,0.5);
        ">
            <h2 style="margin:0; color:white;">🚨 RED ALERT – IMMEDIATE ATTENTION REQUIRED</h2>
            <p style="margin:8px 0 0 0; font-size:1.2em;">
                <b>{alert['label']}</b> detected at <b>{alert['location']}</b><br>
                Confidence: {alert['confidence']*100:.0f}% &nbsp;|&nbsp; Time: {alert['timestamp']}
            </p>
            <p style="margin:5px 0 0 0;">Security / Park Managers: Please investigate this sector now.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Acknowledge & Clear Alert", type="primary"):
        st.session_state.alert_active = False
        st.rerun()
else:
    st.success("✅ System normal – No active threats")

# -------------------------------------------------
# Metrics row
# -------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

total = len(st.session_state.detections)
threats = sum(1 for d in st.session_state.detections if d["is_threat"])
last_hour = sum(
    1 for d in st.session_state.detections
    if datetime.strptime(d["timestamp"], "%Y-%m-%d %H:%M:%S") > datetime.now() - timedelta(hours=1)
)

col1.metric("Total Detections", total)
col2.metric("Threat Detections", threats, delta=None if threats == 0 else "Action needed")
col3.metric("Last Hour", last_hour)
col4.metric("System Status", "ALERT" if st.session_state.alert_active else "Normal")

st.markdown("---")

# -------------------------------------------------
# Main content
# -------------------------------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Recent Detections")

    if demo_mode:
        # Add a new simulated detection occasionally
        if random.random() < 0.4:  # 40% chance each refresh
            det = generate_demo_detection()
            st.session_state.detections.insert(0, det)
            if det["is_threat"]:
                st.session_state.alert_active = True
                st.session_state.last_alert = det
            # Keep only last 50
            st.session_state.detections = st.session_state.detections[:50]

    if st.session_state.detections:
        df = pd.DataFrame(st.session_state.detections)

        # Colour threat rows
        def highlight_threats(row):
            if row["is_threat"]:
                return ["background-color: #4a0000; color: #ffcccc"] * len(row)
            return [""] * len(row)

        display_df = df[["timestamp", "label", "confidence", "location", "is_threat"]].copy()
        display_df["confidence"] = (display_df["confidence"] * 100).round(0).astype(int).astype(str) + "%"
        display_df = display_df.rename(columns={
            "timestamp": "Time",
            "label": "Detected Sound",
            "confidence": "Confidence",
            "location": "Location",
            "is_threat": "Threat"
        })

        st.dataframe(
        display_df,
        use_container_width=True,
        height=420,
        )
    else:
        st.info("No detections yet. Waiting for audio events…")

with right:
    st.subheader("Threat Summary")
    if st.session_state.detections:
        threat_df = pd.DataFrame(st.session_state.detections)
        threat_counts = threat_df[threat_df["is_threat"]]["label"].value_counts()
        if not threat_counts.empty:
            st.bar_chart(threat_counts)
        else:
            st.write("No threat events recorded yet.")
    else:
        st.write("—")

    st.markdown("---")
    st.subheader("How it works")
    st.markdown(
        """
        - Edge model listens continuously  
        - Classifies: **Human / Gunshot / Vehicle**  
        - Smart OOD rejection ignores background noise  
        - Threat → **Red Alert** for security response  
        """
    )

# -------------------------------------------------
# Auto refresh
# -------------------------------------------------
if auto_refresh:
    time.sleep(5)
    st.rerun()