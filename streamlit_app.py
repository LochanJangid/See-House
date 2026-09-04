"""
Golden State Housing Estimator
--------------------------------
Streamlit front end for a California housing price prediction API.

Run:
    pip install streamlit requests pandas
    streamlit run housing_price_estimator.py

Expects a POST endpoint at API_URL that accepts:
    {"inputs": [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]}
and returns:
    {"predictions": [<value in units of $100,000>]}
"""

import requests
import streamlit as st
import pandas as pd

API_URL = "http://localhost:8000/predict"

FEATURE_DEFAULTS = {
    "medinc": 3.87,
    "houseage": 28.0,
    "averooms": 5.43,
    "avebedrms": 1.10,
    "population": 1425.0,
    "aveoccup": 3.07,
    "latitude": 35.63,
    "longitude": -119.57,
}

st.set_page_config(
    page_title="Golden State Housing Estimator",
    page_icon="🏡",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        color: #1B2430;
    }
    .stApp { background-color: #FAFAF8; }
    h1, h2, h3 {
        font-family: 'Fraunces', serif;
        color: #1B2430;
        letter-spacing: -0.01em;
    }
    .subtitle {
        color: #5B6472;
        font-size: 1.05rem;
        margin-top: -0.6rem;
        margin-bottom: 1.5rem;
    }
    .section-label {
        font-weight: 600;
        font-size: 0.88rem;
        color: #1F5673;
        margin: 1.1rem 0 0.35rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #E3E7EC;
    }
    div[data-testid="stForm"] {
        background-color: #FFFFFF;
        border: 1px solid #E3E7EC;
        border-radius: 12px;
        padding: 1.75rem 2rem 1.25rem 2rem;
    }
    .stButton > button,
    div[data-testid="stFormSubmitButton"] button,
    button[kind="formSubmit"],
    button[kind="secondaryFormSubmit"] {
        background-color: #E8871E;
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        padding: 0.55rem 1.6rem;
        font-weight: 600;
    }
    .stButton > button:hover,
    div[data-testid="stFormSubmitButton"] button:hover,
    button[kind="formSubmit"]:hover {
        background-color: #C96F10;
        color: #FFFFFF;
    }
    .result-card {
        background-color: #1F5673;
        border-radius: 12px;
        padding: 2rem 2.25rem;
        color: #FFFFFF;
    }
    .result-card .label { font-size: 0.9rem; color: #C7D6DE; margin-bottom: 0.25rem; }
    .result-card .price {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 2.75rem;
        color: #FFE1AE;
        margin: 0;
    }
    .result-card .footnote { font-size: 0.8rem; color: #9FB4C0; margin-top: 0.75rem; }
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E3E7EC;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### About this estimator")
    st.write(
        "Enter the characteristics of a California block group and get an "
        "estimated median home value from the prediction API running at "
        f"`{API_URL}`."
    )
    if st.button("Fill in typical California values"):
        for key, val in FEATURE_DEFAULTS.items():
            st.session_state[key] = val
        st.rerun()
    with st.expander("What do these fields mean?"):
        st.markdown(
            "- **Median income** — block group's median household income, "
            "in tens of thousands of dollars\n"
            "- **House age** — median age of houses in the block group, in years\n"
            "- **Average rooms** — average rooms per household\n"
            "- **Average bedrooms** — average bedrooms per household\n"
            "- **Population** — total population of the block group\n"
            "- **Average occupancy** — average household size\n"
            "- **Latitude / longitude** — location of the block group"
        )

st.title("Golden State Housing Estimator")
st.markdown(
    '<p class="subtitle">Estimate a median home value for a California block group.</p>',
    unsafe_allow_html=True,
)

with st.form("prediction_form"):
    st.markdown('<div class="section-label">Location</div>', unsafe_allow_html=True)
    loc_col1, loc_col2 = st.columns(2)
    latitude = loc_col1.number_input(
        "Latitude", min_value=32.0, max_value=42.5, step=0.01, format="%.2f",
        key="latitude", value=FEATURE_DEFAULTS["latitude"],
        help="California spans roughly 32.5°N to 42°N.",
    )
    longitude = loc_col2.number_input(
        "Longitude", min_value=-124.5, max_value=-114.0, step=0.01, format="%.2f",
        key="longitude", value=FEATURE_DEFAULTS["longitude"],
        help="California spans roughly -124.5° to -114°.",
    )

    st.markdown('<div class="section-label">Household economics</div>', unsafe_allow_html=True)
    medinc = st.number_input(
        "Median income (tens of thousands of $)", min_value=0.0, max_value=20.0,
        step=0.01, format="%.2f", key="medinc", value=FEATURE_DEFAULTS["medinc"],
        help="E.g. 3.87 ≈ $38,700 median household income.",
    )

    st.markdown('<div class="section-label">Housing stock</div>', unsafe_allow_html=True)
    house_col1, house_col2, house_col3 = st.columns(3)
    houseage = house_col1.number_input(
        "House age (years)", min_value=1.0, max_value=52.0, step=1.0, format="%.0f",
        key="houseage", value=FEATURE_DEFAULTS["houseage"],
    )
    averooms = house_col2.number_input(
        "Average rooms", min_value=0.5, max_value=20.0, step=0.1, format="%.2f",
        key="averooms", value=FEATURE_DEFAULTS["averooms"],
    )
    avebedrms = house_col3.number_input(
        "Average bedrooms", min_value=0.2, max_value=6.0, step=0.1, format="%.2f",
        key="avebedrms", value=FEATURE_DEFAULTS["avebedrms"],
    )

    st.markdown('<div class="section-label">Occupancy</div>', unsafe_allow_html=True)
    occ_col1, occ_col2 = st.columns(2)
    population = occ_col1.number_input(
        "Population", min_value=1.0, max_value=40000.0, step=1.0, format="%.0f",
        key="population", value=FEATURE_DEFAULTS["population"],
    )
    aveoccup = occ_col2.number_input(
        "Average occupancy", min_value=0.5, max_value=20.0, step=0.1, format="%.2f",
        key="aveoccup", value=FEATURE_DEFAULTS["aveoccup"],
        help="Average number of household members.",
    )

    submitted = st.form_submit_button("🏡 Estimate price")

if submitted:
    inputs = [medinc, houseage, averooms, avebedrms, population, aveoccup, latitude, longitude]

    with st.spinner("Talking to the model…"):
        try:
            response = requests.post(API_URL, json={"inputs": inputs})
            response.raise_for_status()
            result = response.json()
            prediction = result.get("prediction")
            value = prediction

            if value is None:
                st.error("The API response didn't include a 'predictions' value.")
            else:
                dollars = float(value) * 100_000

                result_col, map_col = st.columns([3, 2])
                with result_col:
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="label">Estimated median home value</div>
                            <p class="price">${dollars:,.0f}</p>
                            <div class="footnote">
                                Raw model output: {value}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with map_col:
                    st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}), zoom=6)

        except requests.exceptions.ConnectionError:
            st.error(f"Can't reach the prediction API at {API_URL}. Is it running?")
        except requests.exceptions.Timeout:
            st.error("The API took too long to respond. Try again.")
        except requests.exceptions.HTTPError:
            st.error(f"The API returned an error (status {response.status_code}).")
        except ValueError:
            st.error("The API didn't return valid JSON.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")