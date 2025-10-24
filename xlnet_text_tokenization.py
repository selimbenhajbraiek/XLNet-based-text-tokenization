"""
Author: Selim Ben Haj Braiek
Project: FAQ Chatbot using BERT and Hugging Face Transformers
Description: Simple script to clean and tokenize text using XLNet Tokenizer
  
"""

import pandas as pd
import re
from cleantext import clean
from transformers import XLNetTokenizer

# 1 Load dataset
data = pd.read_csv("emotion-labels-train.csv")  # Replace with your file path

# 2 Clean text (remove emojis and punctuation)
def preprocess_text(text):
    text = clean(text, no_emoji=True)
    text = re.sub(r"([^\w\s])", "", text)  # remove punctuation
    return text.strip()

data["clean_text"] = data["text"].apply(preprocess_text)

# 3 Initialize XLNet tokenizer
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")

# 4 Tokenize text samples
tokenized = tokenizer(
    data["clean_text"].tolist(),
    padding="max_length",
    truncation=True,
    max_length=64,
    return_tensors="pt"
)

# 5 Display tokenization results
print("Original Text:", data["clean_text"].iloc[0])
print("Token IDs:", tokenized["input_ids"][0])
print("Decoded Text:", tokenizer.decode(tokenized["input_ids"][0]))

print("\n✅ XLNet tokenization completed successfully!")
