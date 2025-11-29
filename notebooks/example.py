import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import torch
model_name = "bigscience/mt0-small"
test_file = "/Users/karim/Downloads/Датасет_для_обучения_модели_Аскар.csv"
df_test = pd.read_csv(test_file, sep="\t")
df_test.columns = df_test.columns.str.strip()
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.to("cpu")
def detox_tatar(text):
    inputs = tokenizer(f"Detoxify this Tatar text: {text}", return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=128, num_beams=4, no_repeat_ngram_size=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
df_test["detox"] = [detox_tatar(t) for t in tqdm(df_test["toxic"], desc="Detoxifying")]
df_test[["toxic", "detox"]].to_csv("submission.tsv", sep="\t", index=False)