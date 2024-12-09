import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title ='Marvelous AiLegend'
)

st.button("Let's explore")


st.page_link("main.py", label="Home", icon="🏠")
st.page_link("PYTHON_WEBSITE/Backprop.py", label="Backpropagation", icon="🧠")
st.page_link("linkedin.com/in/aafreen-khan-aa9180257", label="LinkedIn", icon="🤩", disabled=True)
st.page_link("marvelousailegend888.com", label="Google", icon="🌎")