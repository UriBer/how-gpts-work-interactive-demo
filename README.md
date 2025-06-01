# 📚 אינטראקציה עם GPT – הדגמה חינוכית | GPT Interactive Learning Demo

מערכת זו מדגימה בצורה אינטראקטיבית את פעולתו של מודל GPT עבור סטודנטים עם רקע בסיסי בפייתון. 
This application interactively demonstrates how GPT works – designed for students with basic Python knowledge.

היא כוללת הדמיה של כל שלב עיקרי: טוקניזציה, embedding, קשב עצמי, השלמת משפט, סימולציית אימון ואיורים גרפיים.
It includes each core stage: tokenization, embeddings, self-attention, sentence generation, training simulation, and visual illustrations.

---
## 🚀 התקנה | Installation Instructions


```bash
pip install -r requirements.txt
python 00-initial-setup.py

```

במקרה של הורדה ראשונית – כל המודלים יישמרו בתיקיית `./models` לשימוש מקומי.
On first run, all models are saved locally in `./models`.

---
## 🚀 הפעלה | Run Instructions

streamlit run Home.py



## 🧭 ניווט בין מסכים | Navigation

| מסך | תיאור בעברית | Description (English) |
|------|----------------|------------------------|
| **מסך 1 – טוקניזציה ו-Embedding** | המרת משפט לטוקנים ווקטורים | Tokenization and embedding demo |
| **מסך 2 – קשב עצמי (Self-Attention)** | מטריצת הקשב בין מילים | Self-attention heatmap visualization |
| **מסך 3 – השלמת משפט** | השלמת משפט בעזרת GPT-2 | Sentence generation using GPT-2 |
| **מסך 4 – סימולציית אימון** | איך GPT "לומד" על בסיס טעויות | Simulated training: how GPT learns |
| **מסך 5 – איורים גרפיים** | תרשימים של positional encoding וקשב | Visual illustrations of positional encoding and attention |

---

## 📁 מבנה התיקיות | Folder Structure

```
interactive-demo-how-gpts-work/
├── Home.py                         # מסך 1 / Page 1
├── pages/
│   ├── 02-self-attention.py       # מסך 2 / Page 2
│   ├── 03-text-generation.py      # מסך 3 / Page 3
│   ├── 04-training-simulation.py  # מסך 4 / Page 4
│   └── 05-visuals.py              # מסך 5 / Page 5
├── 00-initial-setup.py            # הורדת מודלים / Model downloader
├── models/                        # מודלים / Saved models
│   ├── bert-base-multilingual-cased/
│   └── hebrew-gpt-neo/
└── README.md
```

---

## 🌐 שפות | Language Support

- ניתן לעבור בין עברית לאנגלית על בסיס קלט המשתמש.
- App detects preferred language (Hebrew or English).

---

✉️ לשאלות או הצעות: אורי ברמן | Contact: Uri Berman
