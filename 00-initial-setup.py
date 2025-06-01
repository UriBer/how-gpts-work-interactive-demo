# 00-initial-setup.py
# This script downloads the models used in the GPT interactive simulation
# סקריפט זה מוריד את המודלים הדרושים לסימולציה של GPT בעברית ואנגלית

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# ---------- English GPT-2 ----------
# Download English GPT-2 model for examples in English
model_en = "gpt2"
AutoTokenizer.from_pretrained(model_en).save_pretrained("./models/gpt2")
AutoModelForCausalLM.from_pretrained(model_en).save_pretrained("./models/gpt2")

# ---------- Multilingual BERT ----------
# Download multilingual BERT model for tokenization and embeddings
# מודל BERT רב-לשוני לתצוגת אמבדינגים בעברית ובאנגלית
model_bert = "bert-base-multilingual-cased"
AutoTokenizer.from_pretrained(model_bert).save_pretrained("./models/bert-base-multilingual-cased")
AutoModel.from_pretrained(model_bert).save_pretrained("./models/bert-base-multilingual-cased")

# ---------- Hebrew GPT-Neo ----------
# Download Hebrew text generation model fine-tuned on Hebrew data
# מודל ליצירת טקסטים בעברית, מבוסס GPT-Neo
model_he = "Norod78/hebrew-gpt_neo-small"
AutoTokenizer.from_pretrained(model_he).save_pretrained("./models/hebrew-gpt-neo")
AutoModelForCausalLM.from_pretrained(model_he).save_pretrained("./models/hebrew-gpt-neo")

print("✅ All models downloaded successfully into './models'")