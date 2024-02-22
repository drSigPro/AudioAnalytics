import streamlit as st

st.set_page_config(
    page_title="Main",
    page_icon="👋",
)

st.write("# Welcome to Audio Analytics Demo!")

st.markdown(
    """
    The demo consists of sampling (resampling) and quanitzation of audio signals, filtering and then feature extraction
    
    Please select appropriate tab from left pane
    
    
    *Reference: McFee, B., et al. (2015). librosa: Audio and Music Signal Analysis in Python.*
"""
)

