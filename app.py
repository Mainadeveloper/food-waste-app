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

# --- Log user login ---
def log_login(username):
    log_file = "login_log.csv"
    if not os.path.exists(log_file):
        pd.DataFrame(columns=["username", "timestamp"]).to_csv(log_file, index=False)
    df = pd.read_csv(log_file)
    new_entry = pd.DataFrame([[username, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]], columns=["username", "timestamp"])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(log_file, index=False)

# --- User Management ---
def load_users():
    if not os.path.exists("users.csv") or os.path.getsize("users.csv") == 0:
        df = pd.DataFrame(columns=["username", "password", "logins"])
        df.to_csv("users.csv", index=False)
        return df
    df = pd.read_csv("users.csv")
    if "logins" not in df.columns:
        df["logins"] = 0
    return df

def save_user(username, password):
    df = load_users()
    if username in df["username"].values:
        return False
    new_user = pd.DataFrame([[username, password, 0]], columns=["username", "password", "logins"])
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv("users.csv", index=False)
    return True

def increment_login(username):
    df = load_users()
    df.loc[df["username"] == username, "logins"] += 1
    df.to_csv("users.csv", index=False)

# --- Login / Signup Page ---
def login_page():
    with st.container():
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("""
                <div style='background-color: #66bb6a; padding: 20px; border-radius: 10px;'>
                    <h1 style='margin: 0; font-size: 2.5rem; font-weight: bold; color: white;'>
                        Smart Food Planner for Restaurants
                    </h1>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.image("assets/logo.png", width=250)

    choice = st.radio("Choose Option", ["Login", "Sign Up"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if choice == "Login":
        if st.button("Login"):
            df = load_users()
            if ((df["username"] == username) & (df["password"] == password)).any():
                increment_login(username)
                log_login(username)
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
    st.markdown(f"""
        <div style='text-align: center; margin-top: 30px; margin-bottom: 30px;'>
            <h2 style='color: #388e3c; font-size: 2.2rem;'>Welcome, {st.session_state.username}</h2>
            <p style='font-size: 1.2rem; color: #444;'>Plan your daily meal and reduce food waste below.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='color: #2e7d32;'>🍽️ Plan Your Daily Meal</h3>", unsafe_allow_html=True)
    people = st.number_input("Number of people to cook for", min_value=1, step=1)

    selected_foods = st.multiselect("Select Food Types", [
        "Meat", "Cereals", "Rice", "Maize/F", "Wheat/F", 
        "Eggs", "Vegetables", "Milk", "Fruits"])

    food_max_limits_single = {
        "Eggs": 0.3,
        "Milk": 0.5,
        "Fruits": 0.5,
        "Cereals": 0.4,
        "Vegetables": 0.9,
        "Wheat/F": 0.5,  
        "Maize/F": 0.5,  
        "Meat": 0.5,  
        "Rice": 0.5  
    }

    food_base_values = {
        "Meat": 3.5,
        "Cereals": 3.0,
        "Rice": 3.2,
        "Maize/F": 4.145,
        "Wheat/F": 4.145,
        "Eggs": 2.5,
        "Vegetables": 2.86,
        "Milk": 1.5,
        "Fruits": 2.3
    }

    max_food_per_person = 0.9

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

        if len(selected_foods) == 1:
            # --- Use predefined limits ---
            selected = selected_foods[0]
            limit = food_max_limits_single.get(selected, max_food_per_person)
            quantity = round(people * limit, 2)
            st.success(f"Recommended {selected} to Cook per Day: {quantity} kg")
            st.markdown("### 🍱 Breakdown of Food Quantities to Cook")
            st.table(pd.DataFrame([[selected, f"{quantity} kg"]], columns=["Food Type", "Recommended Quantity"]))

        elif len(selected_foods) > 1:
            # --- Use ML Model ---
            for food in food_base_values:
                input_data[food + "(kgs)"] = 0  # Initialize with 0

            df = pd.DataFrame([input_data])
            df['month'] = df['month'].map(month_mapping)
            df['Event Type'] = df['Event Type'].map(event_type_mapping)

            for col in feature_columns:
                if col not in df.columns:
                    df[col] = 0

            df = df[feature_columns]
            prediction = model.predict(df)[0]
            total_food_limit = people * max_food_per_person
            final_prediction = round(min(prediction, total_food_limit), 2)

            st.success(f"Recommended Total Food to Cook per Day: {final_prediction} kg")
            st.markdown("### 🍱 Breakdown of Food Quantities to Cook")

            # Calculate weight sum for selected items
            weight_sum = sum([food_base_values[food] for food in selected_foods])
            table_data = []
            for food in selected_foods:
                portion = round((food_base_values[food] / weight_sum) * final_prediction, 2) if weight_sum > 0 else 0
                table_data.append([food, f"{portion} kg"])

            st.table(pd.DataFrame(table_data, columns=["Food Type", "Recommended Quantity"]))
        else:
            st.warning("Please select at least one food type.")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.show_logout_message = True
        st.rerun()

# --- Logout Page ---
def logout_message_page():
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
