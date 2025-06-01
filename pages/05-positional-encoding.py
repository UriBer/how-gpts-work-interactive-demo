import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# הגדרת העמוד
st.set_page_config(page_title="Visuals: Attention & Positional Encoding", layout="centered")

# שפה מה-session
lang = st.session_state.get("lang", "עברית")

# כותרת
st.title("\U0001F4C8 Attention & Positional Encoding" if lang == "English" else "\U0001F4C8 קשב וקידוד מיקום")

# הסבר ראשוני
st.markdown(
    """
    **Positional Encoding** adds information about the position of each token.
    Below is a visual of sinusoidal encoding used in Transformers.
    """ if lang == "English" else
    """
    **קידוד מיקום** מוסיף למודל מידע על הסדר של המילים במשפט.
    למטה מופיעה המחשה של הקידוד הסינוסואידלי שבשימוש בטרנספורמרים.
    """
)

# פונקציה ליצירת קידוד מיקום סינוסואידלי
@st.cache_data
def get_positional_encoding(max_len=20, d_model=16):
    PE = np.zeros((max_len, d_model))
    position = np.arange(0, max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)
    return PE

pe = get_positional_encoding()

# ציור הגרף
fig, ax = plt.subplots(figsize=(10, 4))
cmap = plt.get_cmap('viridis')
c = ax.pcolormesh(pe.T, cmap=cmap)
ax.set_title("Positional Encoding" if lang == "English" else "קידוד מיקום סינוסואידלי")
ax.set_xlabel("Token Position" if lang == "English" else "מיקום הטוקן")
ax.set_ylabel("Embedding Dimension" if lang == "English" else "מימד embedding")
fig.colorbar(c, ax=ax)
st.pyplot(fig)

# הסבר על Attention Heads
st.markdown(
    """
    **Multi-Head Attention** allows the model to focus on different parts of the sentence at once.
    Below is a simulated example.
    """ if lang == "English" else
    """
    **קשב רב-ראשי (Multi-Head Attention)** מאפשר למודל להסתכל על חלקים שונים במשפט בו זמנית.
    הדוגמה למטה מדמה זאת ויזואלית.
    """
)

# גרף קשב ראשוני לדוגמה (נבנה אקראי)
@st.cache_data
def simulate_attention_matrix(size=6):
    return np.round(np.random.rand(size, size), 2)

attn = simulate_attention_matrix()

# גרף Heatmap
fig2, ax2 = plt.subplots()
c2 = ax2.imshow(attn, cmap='coolwarm')
ax2.set_title("Simulated Attention Weights" if lang == "English" else "משקלי קשב מדומים")
ax2.set_xlabel("Key Tokens")
ax2.set_ylabel("Query Tokens")
fig2.colorbar(c2)
st.pyplot(fig2)

# כפתור חזרה
if st.button("⬅️ Back to Start" if lang == "English" else "⬅️ חזרה להתחלה"):
    st.session_state['navigate'] = False
    st.switch_page("home")
