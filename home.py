import streamlit as st

# Add a title to the homepage
st.title("Welcome to the Homepage")

# Create a button for each page
if st.button("Enter Page 1"):
    import app
elif st.button("Enter Page 2"):
    import ytapp
