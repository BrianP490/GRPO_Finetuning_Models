# app.py
import streamlit as st
from importlib.metadata import version
import numpy, torch, os, pandas
import tensorflow

library = ["numpy", "torch", "tensorflow", "streamlit", "pandas"]

for lib in library:
    st.write(f"{lib} version: {version(lib)}")

# Set basic page configuration (optional, but good for wider layouts)
st.set_page_config(
    page_title="My First Streamlit App",
    page_icon="👋",
    layout="centered", # or "wide"
    initial_sidebar_state="auto"
)

st.title("Hello, Streamlit World! 👋")

# https://docs.streamlit.io/develop/api-reference/widgets/st.text_area
text_block = st.text_area(label="Enter your text to classify if it is SPAM or NOT SPAM", placeholder ="ConGratulations!!!1 You won $1.000. Click the link beelow to claime you're Prize.!")

st.write("""
This is a simple Streamlit application.
You can interact with the widgets below!
""")

# Text input widget
user_name = st.text_input("What's your name?", "Guest")
st.write(f"Nice to meet you, {user_name}!")

# Slider widget
age = st.slider("How old are you?", 0, 100, 25)
st.write(f"You are {age} years old.")

# Button widget
if st.button("Click Me!"):
    st.success("You clicked the button!")

st.markdown("---")
st.caption("This app was created with Streamlit.")

# SIDEBAR: https://docs.streamlit.io/develop/api-reference/layout/st.sidebar
# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)

# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0)
)

# Using "with" notation
with st.sidebar:
    add_radio = st.radio(
        "Choose a shipping method",
        ("Standard (5-15 days)", "Express (2-5 days)")
    )