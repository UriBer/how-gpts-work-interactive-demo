# 03-text-generation.py
# Hebrew text generation using Norod78/hebrew-gpt_neo-small
# יצירת המשך טקסט בעברית באמצעות המודל Norod78/hebrew-gpt_neo-small

import os
import streamlit as st
from transformers import pipeline

# Prevent tokenizer fork error
# מניעת קריסות בסביבות עם fork (בעיית tokenizer מקבילים)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# הגדרת תצוגת העמוד
st.set_page_config(page_title="Text Generation - Hebrew GPT-Neo", layout="centered")

# שליפת שפת הממשק מה-session
lang = st.session_state.get("lang", "עברית")

# משפט ברירת מחדל
user_sentence = st.session_state.get(
    "user_sentence",
    "המהפכה של GPT" if lang == "עברית" else "The GPT revolution"
)

# כותרת
st.title("🧠 Text Generation" if lang == "English" else "🧠 השלמת משפט בעברית")

# תיאור
st.markdown(
    "Generate Hebrew sentence completions using a Hebrew GPT-Neo model." if lang == "English" else
    "יצירת המשך למשפט בעברית באמצעות המודל Norod78/hebrew-gpt_neo-small."
)

# קלט מהמשתמש
user_input = st.text_input(
    "Enter a sentence to complete:" if lang == "English" else "הכנס התחלה של משפט להשלמה:",
    value=user_sentence
)

# טעינת הפייפליין פעם אחת בלבד במטמון
@st.cache_resource
def load_generator():
    return pipeline(
        "text-generation",
        model="./models/hebrew-gpt-neo",
        tokenizer="./models/hebrew-gpt-neo"
    )

generator = load_generator()

# לחצן הפעלה
if st.button("Generate!" if lang == "English" else "השלם משפט"):
    with st.spinner("Generating..." if lang == "English" else "מייצר המשך למשפט..."):
        result = generator(
            user_input,
            max_new_tokens=60,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            num_return_sequences=1
        )[0]['generated_text']

    # תצוגת התוצאה
    st.subheader("➡️ Result" if lang == "English" else "➡️ תוצאה")
    st.success(result)

    # הסבר מה קרה מאחורי הקלעים
    st.markdown("""
    **Explanation (English):**  
    This page uses the `Norod78/hebrew-gpt_neo-small` model to generate Hebrew text based on a sentence prefix.  
    The model was fine-tuned specifically for Hebrew and can generate creative sentence continuations.

    **הסבר (עברית):**  
    העמוד הזה משתמש במודל `Norod78/hebrew-gpt_neo-small` שנאמן במיוחד לשפה העברית.  
    הוא מקבל התחלה של משפט ויוצר השלמה טבעית, באמצעות sampling עם טמפרטורה גבוהה ומסנני top-k/top-p.
    """)

# לחצן חזרה למסך הבית
if st.button("⬅️ Start Over" if lang == "English" else "⬅️ התחל מההתחלה"):
    st.session_state['navigate'] = False
    st.switch_page("home")