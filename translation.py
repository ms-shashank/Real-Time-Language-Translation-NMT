from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load pre-trained models for specific languages
tokenizer_t5 = AutoTokenizer.from_pretrained("t5-base")
model_t5 = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

# Use Helsinki-NLP for English to Spanish
tokenizer_es = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
model_es = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")

def translate_with_t5(input_text, src_lang, tgt_lang):
    task_prefix = f"translate {src_lang} to {tgt_lang}: "
    formatted_input = task_prefix + input_text

    inputs = tokenizer_t5(
        formatted_input, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    outputs = model_t5.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
    return tokenizer_t5.decode(outputs[0], skip_special_tokens=True)

def translate_to_spanish(input_text):
    inputs = tokenizer_es(input_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model_es.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
    return tokenizer_es.decode(outputs[0], skip_special_tokens=True)

# Example: Translate into multiple languages
source_text = "How are you? I am fine."
translations = {
    "French": translate_with_t5(source_text, "English", "French"),
    "German": translate_with_t5(source_text, "English", "German"),
    "Spanish": translate_to_spanish(source_text),  # Use specialized model
}

# Display translations
for lang, translated_text in translations.items():
    print(f"Translated to {lang}: {translated_text}")
