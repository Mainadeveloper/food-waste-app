import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# --- Load ML model and encoders ---
model = joblib.load("model/rf_model.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")
month_mapping = joblib.load("model/month_mapping.pkl")
event_type_mapping = joblib.load("model/event_type_mapping.pkl")

# --- Streamlit Config ---
st.set_page_config(page_title="Smart Food Planner", layout="wide")

# --- Load Custom CSS ---
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- User Management ---
def load_users():
    if not os.path.exists("users.csv") or os.path.getsize("users.csv") == 0:
        df = pd.DataFrame(columns=["username", "password"])
        df.to_csv("users.csv", index=False)
        return df
    return pd.read_csv("users.csv")

def save_user(username, password):
    df = load_users()
    if username in df["username"].values:
        return False
    new_user = pd.DataFrame([[username, password]], columns=["username", "password"])
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv("users.csv", index=False)
    return True

# --- Login / Signup Page ---
def login_page():
    with st.container():
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(
                """
                <div style='background-color: #66bb6a; padding: 20px; border-radius: 10px;'>
                    <h1 style='margin: 0; font-size: 2.5rem; font-weight: bold; color: white;'>
                        Smart Food Planner for Restaurants
                    </h1>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.image("assets/logo.png", width=250)

    with st.container():
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(
                """
                <h2 style='font-weight: bold; margin-bottom: 0.5rem;'>Let's save food</h2>
                <p style='font-size: 1.1rem; margin-bottom: 1.5rem;'>
                Food waste is a growing concern, especially in restaurants where planning can make all the difference.<br>
                As food providers, we have the power to reduce waste with smarter, data-driven decisions.<br>
                The solution starts in the kitchen—with the right tools and insights, we can serve just enough and waste far less.
                </p>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.image("assets/logo4.png", width=300)

    choice = st.radio("Choose Option", ["Login", "Sign Up"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if choice == "Login":
        if st.button("Login"):
            df = load_users()
            if ((df["username"] == username) & (df["password"] == password)).any():
                st.success("Logged in successfully")
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.show_logout_message = False
                st.rerun()
            else:
                st.error("Invalid credentials")
    else:
        if st.button("Sign Up"):
            if save_user(username, password):
                st.success("Account created. Please login.")
            else:
                st.error("Username already exists.")

# --- Dashboard Page ---
def dashboard_page():
    st.markdown(
        f"""
        <div style='text-align: center; margin-top: 30px; margin-bottom: 30px;'>
            <h2 style='color: #388e3c; font-size: 2.2rem;'>Welcome, {st.session_state.username}</h2>
            <p style='font-size: 1.2rem; color: #444;'>Plan your meal and reduce food waste below.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h3 style='color: #2e7d32;'>🍽️ Plan Your Meal</h3>", unsafe_allow_html=True)
    people = st.number_input("Number of people to cook for", min_value=1, step=1)

    selected_foods = st.multiselect("Select Food Types", [
        "Meat(kgs)", "Cereals(kgs)", "Rice(kgs)", "Maize/F(kgs)", "Wheat/F(kgs)", 
        "Eggs(kgs)", "Vegetables(kgs)", "Milk(litres)", "Fruits(kgs)"
    ])

    if st.button("Predict Ideal Food Quantities"):
        current_month = datetime.now().month

        input_data = {
            "Total Customers": people,
            "Event Type": "Regular",
            "month": current_month,
            "B/F": people * 0.3,
            "Lunch": people * 0.5,
            "supper": people * 0.2,
            "Food cooked": people * 1.0,
            "Food Consumed": people * 0.95
        }

        food_base_values = {
            "Meat(kgs)": 3.5,
            "Cereals(kgs)": 3.0,
            "Rice(kgs)": 3.2,
            "Maize/F(kgs)": 4.145,
            "Wheat/F(kgs)": 4.145,
            "Eggs(kgs)": 2.5,
            "Vegetables(kgs)": 2.86,
            "Milk(litres)": 1.5,
            "Fruits(kgs)": 2.3
        }

        total_custom_weight = 0
        base_food_values = {}

        for food, value in food_base_values.items():
            if food in selected_foods:
                input_data[food] = value
                base_food_values[food] = value
                total_custom_weight += value
            else:
                input_data[food] = 0

        df = pd.DataFrame([input_data])
        df['month'] = df['month'].map(month_mapping)
        df['Event Type'] = df['Event Type'].map(event_type_mapping)

        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[feature_columns]
        prediction = model.predict(df)[0]
        rounded_pred = round(prediction, 2)

        st.success(f"Recommended Total Food to Cook: {rounded_pred} kg/ltr")

        st.markdown("### 🍱 Breakdown of Food Quantities to Cook")
        table_data = []
        for food, base_value in base_food_values.items():
            portion = round((base_value / total_custom_weight) * rounded_pred, 2)
            table_data.append([food, f"{portion} kg/ltr"])

        st.table(pd.DataFrame(table_data, columns=["Food Type", "Recommended Quantity"]))

        st.markdown("### Input Summary")
        display_df = df.drop(columns=["B/F", "Lunch", "supper"], errors='ignore')
        st.dataframe(display_df)

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.show_logout_message = True
        st.rerun()


# --- Logout Page ---
def logout_message_page():
    # Display image and text only on the logout page
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image("assets/logo6.png", width=300)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("""
        <div style='display: flex; justify-content: center; align-items: center; height: 50vh;'>
            <h1 style='color: #66bb6a; font-family: "Permanent Marker", cursive; font-size: 3em;'>
                Plan Smart. Serve Right. Waste Less.
            </h1>
        </div>
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Permanent+Marker&display=swap');
        </style>
    """, unsafe_allow_html=True)
     # Display image at bottom
    if st.button("Return to Login"):
        st.session_state.show_logout_message = False
        st.rerun()

# --- Routing ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "show_logout_message" not in st.session_state:
    st.session_state.show_logout_message = False

if st.session_state.logged_in:
    dashboard_page()
elif st.session_state.show_logout_message:
    logout_message_page()
else:
    login_page()
