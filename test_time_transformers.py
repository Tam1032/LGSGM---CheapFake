from transformers import AutoTokenizer
from intel_extension_for_transformers.transformers import AutoModelForSeq2SeqLM
import time

tokenizer = AutoTokenizer.from_pretrained("lizhuang144/flan-t5-base-VG-factual-sg")
model = AutoModelForSeq2SeqLM.from_pretrained("lizhuang144/flan-t5-base-VG-factual-sg")

start = time.time()
text = tokenizer(
    "a boy and a girl holding hands while sitting on a shore.",
    max_length=200,
    return_tensors="pt",
    truncation=True
)

generated_ids = model.generate(
    text["input_ids"],
    attention_mask=text["attention_mask"],
    use_cache=True,
    decoder_start_token_id=tokenizer.pad_token_id,
    num_beams=1,
    max_length=200,
    early_stopping=True
)
end = time.time()
print(end-start)

print(tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
# Output: `( pigs, is, 2), (bags, on back of, pigs), (bags, is, 2), (pigs, fly on, sky )`