import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModel  # טוען את המודל ואת ה-tokenizer (כלי לפירוק טקסט לטוקנים)
import torch  # ספריית עיבוד מתמטי למודלים של למידה עמוקה
import numpy as np

# הגדרת פרמטרים כלליים לאפליקציה
st.set_page_config(page_title="GPT Interactive Demo", layout="centered")

# בוחר שפה (עברית או אנגלית) ושומר בזיכרון זמני (session)
target_lang = st.radio("Select language | בחר שפה", ["English", "עברית"])
st.session_state['lang'] = target_lang

# טוען את המודל וה-tokenizer פעם אחת ומטמון אותם לשימוש עתידי (חוסך זמן ועומס)
@st.cache_resource
def load_tokenizer_and_model():
    tokenizer = AutoTokenizer.from_pretrained("./models/bert-base-multilingual-cased")
    model = AutoModel.from_pretrained("./models/bert-base-multilingual-cased")
    return tokenizer, model

tokenizer, model = load_tokenizer_and_model()

# הגדרת משפט ברירת מחדל בהתאם לשפה שנבחרה
default_sentence = "The GPT revolution" if target_lang == "English" else "המהפכה של GPT"

# הגדרת משתנה לזיהוי האם בוצעה קפיצה אוטומטית
if 'navigate' not in st.session_state:
    st.session_state['navigate'] = False

# כותרת לאפליקציה
st.title("📚 Step 1: How GPT Works" if target_lang == "English" else "📚 שלב ראשון: איך GPT עובד")

# תיאור כללי קצר על השלב הראשון
st.markdown(
    "Let's explore how GPT processes language step-by-step." if target_lang == "English" else
    "בואו נבין יחד איך GPT מעבד שפה שלב אחרי שלב."
)

# שדה להזנת משפט שנשמר ב-session
title_text = "✍️ Enter your sentence:" if target_lang == "English" else "✍️ הכנס משפט לבחירה:"
st.subheader(title_text)
user_sentence = st.text_input("Sentence" if target_lang == "English" else "משפט", value=default_sentence)
st.session_state['user_sentence'] = user_sentence

# מעבר אוטומטי למסך 2 אם זו הפעם הראשונה
if user_sentence and not st.session_state['navigate']:
    st.session_state['navigate'] = True
    st.switch_page("pages/02-self-attention.py")

# Tokenize and get embeddings
# Convert the input text into tokens (numbers) that the model can process
# המר את הטקסט לטוקנים (מזהים מספריים) עבור המודל
inputs = tokenizer(user_sentence, return_tensors="pt", add_special_tokens=True)

# pt = PyTorch tensors → מערכים מספריים המותאמים ללמידה עמוקה
with torch.no_grad():  # מצב ללא חישובי גרדיאנטים (חוסך משאבים בהסקה)
    outputs = model(**inputs)  # העבר את הקלט דרך המודל וקבל את הפלט

# Extract tokens and embeddings
# קבל את מזהי הטוקנים המקוריים מתוך הקלט
# ולאחר מכן המר אותם לטקסט קריא (מחרוזות טקסט)
token_ids = inputs["input_ids"][0]
tokens = tokenizer.convert_ids_to_tokens(token_ids)

# המרה של הפלט מהשכבה האחרונה לוקטורים נומריים בעזרת numpy
# last_hidden_state מייצג את ה-embedding של כל טוקן בפלט
embeddings = outputs.last_hidden_state[0].numpy()

# כותרת לקטגוריית ההטמעות
st.subheader("🧠 Embeddings" if target_lang == "English" else "🧠 ייצוגים (Embedding)")

# הסבר מילולי על הטוקנים והמבנה של המודל
if target_lang == "English":
    st.markdown("""
    Each token (word or part of a word) is represented as a vector of numbers. We show only 3 dimensions for simplicity.

    **Special tokens:**
    - `[CLS]` — Classification token: summarizes the entire sentence for downstream tasks.
    - `[SEP]` — Separator token: marks the end of input or separates segments.

    **Tokens with `##` prefix** are sub-word units — they indicate that the token is part of a word continued from the previous token.
    """)
else:
    st.markdown("""
    כל טוקן (יחידת טקסט) מיוצג כוקטור מספרי שמתאר את משמעותו.
    לצורך הפשטה אנו מציגים כאן רק 3 ממדים מתוך הוקטור המלא.

    **עוגני טוקנים מיוחדים:**
    - `[CLS]` — טוקן שמרכז את כל משמעות המשפט (Classification token)
    - `[SEP]` — טוקן שמסמן סוף קלט או מפריד בין משפטים

    **טוקנים שמתחילים ב־`##`** הם חלקי מילים. זה אומר שהטוקן הוא המשך של מילה שהתחילה בטוקן קודם.
    """)

# יצירת טבלת פלט המציגה את שלושת הממדים הראשונים מכל embedding
columns = ["Dim 1", "Dim 2", "Dim 3"] if target_lang == "English" else ["ממד 1", "ממד 2", "ממד 3"]
emb_df = pd.DataFrame(embeddings[:, :3], columns=columns)  # יצירת DataFrame מתוך רשימת הוקטורים
emb_df.insert(0, "Token" if target_lang == "English" else "טוקן", tokens)  # הוספת עמודת טוקנים בהתחלה

# הצגת הטבלה למשתמש
st.dataframe(emb_df)

# כפתור מעבר ידני לשלב הבא
if st.button("➡️ Next: Self-Attention" if target_lang == "English" else "➡️ המשך לשלב הקשב העצמי"):
    st.switch_page("pages/02-self-attention.py")