import streamlit as st
from datetime import datetime


name = st.text_input("What's your name?","Rishabh")
date = st.date_input("Choose a date", datetime.now())
st.markdown(f"## Hello {name}!\nThe date is {date.strftime('%Y-%M-%d')}")


