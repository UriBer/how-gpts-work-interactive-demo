import streamlit as st
import pandas as pd
import numpy as np
import torch  # PyTorch for tensor operations and model execution
from transformers import AutoTokenizer, AutoModel  # huggingface tools for loading tokenizer and model
import matplotlib.pyplot as plt  # for plotting the heatmap

# קביעת תצוגת עמוד
title = "GPT Self-Attention Demo"
st.set_page_config(page_title=title, layout="centered")

# שימוש בשפה מה-session הקודם
lang = st.session_state.get("lang", "עברית")

# טוען מודל ו-tokenizer מהתיקייה המקומית (נדרש כדי למנוע הורדה מהאינטרנט)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("./models/bert-base-multilingual-cased")
    model = AutoModel.from_pretrained(
        "./models/bert-base-multilingual-cased",
        output_attentions=True  # נבקש מהמודל להחזיר גם את שכבות הקשב (attention layers)
    )
    return tokenizer, model

tokenizer, model = load_model()

# שימוש במשפט מה-session הקודם או הגדרה לפי ברירת מחדל
user_sentence = st.session_state.get("user_sentence", "המהפכה של GPT" if lang == "עברית" else "The GPT revolution")

# Tokenize the sentence to convert it to tensor form for the model
# הפיכת הטקסט למזהים מספריים (טוקנים) שעוברים למודל
inputs = tokenizer(user_sentence, return_tensors="pt", add_special_tokens=True)

# הרצת המודל עם הקלט וחילוץ שכבות הקשב (attention)
# torch.no_grad() חוסך משאבים ע"י מניעת חישובי גרדיאנט
with torch.no_grad():
    outputs = model(**inputs)

# המרת המזהים המספריים (input_ids) בחזרה למחרוזות טקסט
# לדוגמה: 101 → [CLS], 102 → [SEP]
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# קבלת מטריצת הקשב מהשכבה האחרונה, head מספר 0
# outputs.attentions = list of tensors [layer][batch][head][token_from][token_to]
# כאן נבחר: layer אחרון, batch ראשון, head ראשון
attentions = outputs.attentions[-1]  # shape: [1, num_heads, seq_len, seq_len]
attn_matrix = attentions[0, 0].numpy()  # Convert tensor to NumPy matrix for visualization

# כותרת
st.subheader("\U0001F441\ufe0f Self-Attention" if lang == "English" else "\U0001F441\ufe0f מנגנון הקשב העצמי (Self-Attention)")

# הסבר טקסטואלי על מטריצת הקשב
st.markdown(
    """The chart below shows how much attention each word gives to the others.\nDarker colors = stronger influence.""" if lang == "English" else
    """התרשים למטה מציג כמה כל מילה מתחשבת בשאר המילים במשפט.\nצבע כהה יותר = השפעה חזקה יותר."""
)

# יצירת Heatmap עם matplotlib
# הצירים מציגים את שמות הטוקנים, והצבעים מציגים את חוזק הקשב ביניהם
fig, ax = plt.subplots()
cax = ax.matshow(attn_matrix, cmap='Blues')  # מצייר את מטריצת הקשב בצבעים
fig.colorbar(cax)  # מוסיף סרגל צבעים

# כיוונון הצירים להצגת שמות הטוקנים
ax.set_xticks(range(len(tokens)))
ax.set_yticks(range(len(tokens)))
ax.set_xticklabels(tokens, rotation=45)
ax.set_yticklabels(tokens)
ax.set_xlabel("Attention To" if lang == "English" else "קשב אל")
ax.set_ylabel("From Token" if lang == "English" else "מהטוקן")

# תצוגת הגרף באפליקציה
st.pyplot(fig)

# רמז למשתמש להמשיך לשלב הבא
st.info("⬅️ Next: Sentence completion (generation)" if lang == "English" else "⬅️ הבא: השלמת משפט עם המודל")

# כפתור מעבר ידני למסך 3
if st.button("🚀 Next: Text Generation" if lang == "English" else "🚀 המשך להשלמת משפט"):
    st.switch_page("pages/03-text-generation.py")
