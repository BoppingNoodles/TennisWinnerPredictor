import streamlit as st
import pandas as pd
import os
import joblib


artifacts = joblib.load('tennis_model.pkl')
model = artifacts['model']
numeric_cols = artifacts['numeric_cols']
feature_order = artifacts.get('feature_order', numeric_cols)
df = pd.read_pickle('cleaned_df.pkl')


st.title('Tennis Match Predictor. Who is more likely to win?')

# Get players' names in a dropdown menu so user can select

player_names = sorted(df['player'].unique())
player_option1 = st.selectbox('Search for a player:', player_names, index = None, placeholder = "Player")
player_option2 = st.selectbox('Search for a player', player_names, index = None, placeholder = "Player")
