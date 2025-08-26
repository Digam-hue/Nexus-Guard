import streamlit as st, pandas as pd, requests

st.set_page_config(page_title="Fraud Detector", layout="wide")
st.title("🚨 Fraud Risk Dashboard")

tab1, tab2 = st.tabs(["Upload CSV", "Single Transaction"])

with tab1:
    f = st.file_uploader("Upload transactions CSV", type=["csv"])
    if f:
        df = pd.read_csv(f)
        st.write("Preview:", df.head())
        try:
            if st.button("Score file"):
                resp = requests.post("http://localhost:8000/score_csv", files={"file": ("data.csv", df.to_csv(index=False), "text/csv")})
                res = resp.json()
                st.write("Top 10 scored preview")
                st.dataframe(pd.DataFrame(res["preview"]))
        except:
            st.write('error')
with tab2:
    amt = st.number_input("Amount", 0.0, 100000.0, 1499.0)
    
    category = st.text_input("Category", "grocery_pos")
    hour = st.number_input("Hour (0-23)", 0, 23, 10)
    city = st.text_input("City", "Mumbai")
    state = st.text_input("State", "MH")
    lat = st.number_input("lat", -90.0, 90.0, 19.07)
    lon = st.number_input("long", -180.0, 180.0, 72.87)
    merch_lat = st.number_input("merch_lat", -90.0, 90.0, 19.00)
    merch_long = st.number_input("merch_long", -180.0, 180.0, 72.80)
    if st.button("Score transaction"):
        payload = {"amt": amt, "category": category, "hour": int(hour), "city": city, "state": state,
                   "lat": float(lat), "long": float(lon), "merch_lat": float(merch_lat), "merch_long": float(merch_long)}
        res = requests.post("http://localhost:8000/score", json=payload).json()
        st.metric("Risk", f"{res['risk']:.2f}")
        st.write("Reasons:", res["reasons"])
'''
# in app/streamlit_app.py
resp = requests.post("http://0.0.0.0:8000/score_csv", files={"file": ("data.csv", df.to_csv(index=False), "text/csv")})
# and line 31
res = requests.post("http://0.0.0.0:8000/score", json=payload).json()
'''