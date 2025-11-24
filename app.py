import streamlit as st
import pandas as pd
import os
import joblib



@st.cache_resource
def load_model_and_data():
    artifacts = joblib.load('tennis_model.pkl')
    df = pd.read_pickle('cleaned_df.pkl')
    return artifacts, df

def predict_winner(playerA, playerB, artifacts, df):
    """
    Predict winner between two tennis players
    """
    model = artifacts['model']
    numeric_cols = artifacts['numeric_cols']
    feature_order = artifacts['feature_order']

    playerA_data = df[df['player'] == playerA]
    playerB_data = df[df['player'] == playerB]

    playerA_stats = playerA_data.iloc[-1]
    playerB_stats = playerB_data.iloc[-1]

    input_data = {}

    for col in numeric_cols:
        if col.startswith('player_'):
            stat_name = col.replace('player_', '')
            if stat_name in playerA_stats:
                    input_data[col] = playerA_stats[stat_name]

    input_data['rank_difference'] = playerA_stats.get('rank', 0) - playerB_stats.get('rank', 0)
    input_data['age_difference'] = playerA_stats.get('age', 0) - playerB_stats.get('age', 0)
    input_data['rank_point_difference'] = playerA_stats.get('rank_points', 0) - playerB_stats.get('rank_points', 0)
    input_data['best_of'] = 3  # Default to best of 3

    for col in feature_order:
        if col not in input_data:
            # Check if this column exists in the player stats
            if col in playerA_stats.index:
                input_data[col] = playerA_stats[col]
            else:
                # If column doesn't exist, set to 0 or appropriate default
                input_data[col] = 0

    X = pd.DataFrame([input_data])
    X = X[feature_order]

    prediction = model.predict(X)[0]
    prediction_proba = model.predict_proba(X)[0]

    if prediction == 1:
     winner = playerA
     winner_prob = prediction_proba[1]
    else:
        winner = playerB
        winner_prob = prediction_proba[0]
    return {
        'playerA': playerA,
        'playerB': playerB,
        'predicted_winner': winner,
        'confidence': round(winner_prob * 100,2),
        'playerA_win_probability': round(prediction_proba[1] * 100,2),
        'playerB_win_probability': round(prediction_proba[0] * 100,2)
    }


artifacts, df = load_model_and_data()
st.title(':green[Tennis Match Predictor🎾]')
st.markdown('### Predict the winner between two tennis players using machine learning')


# Get players' names in a dropdown menu so user can select
player_names = sorted(df['player'].unique())
player_option1 = st.selectbox('Search for a player:', player_names, index = None, placeholder = "Select Player 1")
player_option2 = st.selectbox('Search for a player', player_names, index = None, placeholder = "Select Player 2")


if st.button(':green[Predict Winner!]'):
    if player_option1 is None or player_option2 is None:
        st.warning('Invalid inputs!')
    elif (player_option1 == player_option2) or (player_option2 == player_option1):
        st.warning('Please select 2 different players!')
    else:
        with st.spinner('Predicting...'):
            result = predict_winner(player_option1, player_option2, artifacts, df)

        if('error' in result):
            st.error(result['error'])
        else:
            st.success('Prediction Complete')

            st.markdown ('### Match Prediction')
            st.markdown(f"**{result['playerA']}** vs ** {result['playerB']}**")
            st.markdown('---')

            st.markdown(f"## Predicted Winner: :green[{result['predicted_winner']}] 🏆")
            st.markdown(f"### Confidence: :blue[{result['confidence']:.2f}%]")
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label=f"{result['playerA']} Win Probability",
                    value=f"{result['playerA_win_probability']:.2f}%"
                )
            with col2:
                st.metric(
                    label=f"{result['playerB']} Win Probability",
                    value=f"{result['playerB_win_probability']:.2f}%"
                )



    


