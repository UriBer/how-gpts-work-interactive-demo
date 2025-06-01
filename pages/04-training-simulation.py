import streamlit as st
import random
import time

# הגדרת תצוגת העמוד
st.set_page_config(page_title="Training Simulation", layout="centered")

# קבלת שפה מה-session
lang = st.session_state.get("lang", "עברית")

# כותרת
st.title("\U0001F393 How GPT Learns" if lang == "English" else "\U0001F393 איך GPT מתאמן ולומד")

# תיאור כללי
st.markdown(
    """
    This simulation shows how GPT learns to predict the next word by training on millions of sentences.
    """ if lang == "English" else
    """
    סימולציה זו ממחישה איך GPT מתאמן על ניבוי המילה הבאה בעזרת מיליוני משפטים.
    """
)

# משפטי אימון לדוגמה
examples = [
    ["The", "GPT", "revolution", "is", "here"],
    ["Artificial", "intelligence", "is", "changing", "the", "world"],
    ["Deep", "learning", "models", "are", "powerful"],
    ["GPT", "can", "generate", "human-like", "text"]
]

# בחירת משפט
sentence = random.choice(examples)

# הצגת המשפט
st.markdown("**Training on sentence:**" if lang == "English" else "**מתאמן על המשפט:**")
st.code(" ".join(sentence))

# נבצע תהליך הדרגתי של ניבוי מילה הבאה
st.markdown("**Learning process:**" if lang == "English" else "**תהליך הלמידה:**")
progress_placeholder = st.empty()

# ננבא את המילה הבאה בכל שלב ונשווה לתשובה האמיתית
log = ""
for i in range(len(sentence) - 1):
    context = sentence[:i+1]
    true_next = sentence[i+1]
    # נבנה רשימת "ניחושים" שרירותיים (הדמיה בלבד)
    candidates = random.sample([w for w in sentence if w != true_next], 2) + [true_next]
    random.shuffle(candidates)
    guess = random.choice(candidates)
    correct = (guess == true_next)

    log += f"Input: {' '.join(context)}\n"
    log += f"Prediction: {guess} {'✅' if correct else '❌'} (True: {true_next})\n\n"
    progress_placeholder.code(log)
    time.sleep(0.8)

# סיכום
st.success("Model improves over time by comparing predictions to real next words!" if lang == "English" else "המודל משתפר עם הזמן על ידי השוואת ניבויים למילים האמיתיות!")

# כפתור חזרה
if st.button("⬅️ Back to Generation" if lang == "English" else "⬅️ חזרה להשלמת משפט"):
    st.switch_page("pages/03-text-generation.py")
# כפתור חזרה לתחילת האפליקציה
if st.button("⬅️ Start Over" if lang == "English" else "⬅️ התחל מההתחלה"):
    st.session_state['navigate'] = False
    st.switch_page("home")