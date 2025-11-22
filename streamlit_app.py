import streamlit as st
import pandas as pd
import os

df = pd.read_csv('cleaned_df.pkl')

st.title('Tennis Match Predictor')

# Get players' names in a dropdown menu so user can select

player_names = sorted(df['player'].unique())
player_option = st.selectbox('Choose a player:', player_names)