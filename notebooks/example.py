import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
test_file = "test.tsv"
df_test = pd.read_csv(test_file, sep="\t")
df_test.columns = df_test.columns.str.strip()
assert "ID" in df_test.columns
assert "tat_toxic" in df_test.columns
model_name = "bigscience/mt0-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.to("cpu")
def detox_tatar(text):
    inputs = tokenizer(f"Detoxify this Tatar text: {text}", return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=128, num_beams=4, no_repeat_ngram_size=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
df_test["tat_detox1"] = [detox_tatar(t) for t in tqdm(df_test["tat_toxic"], desc="Detoxifying")]
df_test[["ID", "tat_toxic", "tat_detox1"]].to_csv("submission.tsv", sep="\t", index=False)
