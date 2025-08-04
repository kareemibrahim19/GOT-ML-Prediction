import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load model and preprocessing objects
model = joblib.load("got_best_model.pkl")
scaler = joblib.load("got_scaler.pkl")
pca = joblib.load("got_pca.pkl")
features = joblib.load("scaler_features.pkl")

st.set_page_config(page_title="GOT Character Survival Prediction", layout="wide")

# Styling for light and dark modes
st.markdown("""
    <style>
        body { color: black; background-color: white; }
        .stTextInput > label, .stSelectbox > label {
            font-weight: bold;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🔮 Game of Thrones - Character Survival Predictor")
st.markdown("---")

# Sidebar input container
with st.sidebar:
    st.header("🧬 Character Traits Input")

    def binary_input(label):
        return 1 if st.selectbox(label, ("No", "Yes")) == "Yes" else 0

    male = binary_input("Is Male")
    deadRelations = st.slider("Number of Dead Relations", 0, 10, 0)
    popularity = st.slider("Popularity Score", 0.0, 100.0, 10.0)
    boolDeadRelations = binary_input("Has Dead Relations")
    age = st.slider("Age", 0, 120, 25)

    male_alive = 1 if male == 1 else 0
    female_alive = 1 if male == 0 else 0

    input_data = np.array([[male, deadRelations, popularity, boolDeadRelations,
                            age, male_alive, female_alive]])

    df_input = np.zeros((1, len(features)))
    for idx, feature in enumerate(features):
        if feature in ['male', 'numDeadRelations', 'popularity', 'boolDeadRelations',
                       'age', 'male_alive', 'female_alive']:
            feature_map = {
                'male': male,
                'numDeadRelations': deadRelations,
                'popularity': popularity,
                'boolDeadRelations': boolDeadRelations,
                'age': age,
                'male_alive': male_alive,
                'female_alive': female_alive
            }
            df_input[0, idx] = feature_map[feature]

# Main area
st.header("📈 Prediction")

if st.button("🔍 Predict Survival"):
    x_scaled = scaler.transform(df_input)
    x_pca = pca.transform(x_scaled)
    prediction = model.predict(x_pca)[0]
    prob = model.predict_proba(x_pca)[0]

    survive_prob = prob[1]
    die_prob = prob[0]

    result_text = f"🟢 Survives ({survive_prob * 100:.2f}%)" if prediction == 1 else f"🔴 Dies ({die_prob * 100:.2f}%)"

    st.markdown(f"## Result: {result_text}")

    # Bar chart of prediction probability
    fig, ax = plt.subplots()
    bars = ax.bar(['Dies', 'Survives'], prob, color=['red', 'green'])
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)
    ax.set_title("Survival Probability")

    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height * 100:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    st.pyplot(fig)

else:
    st.info("👈 Please enter the character details in the sidebar and click Predict")