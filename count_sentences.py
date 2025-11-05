from datasets import load_dataset
import re

# Load WikiText-103 raw
dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")

# Simple sentence counter (splits on . ! ?)
def count_sentences(text):
    # Split on sentence boundaries
    sentences = re.split(r'[.!?]+', text)
    # Filter out empty strings and very short fragments
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return len(sentences)

# Count sentences in first 10000 examples to estimate
sample_size = 10000
total_sentences = 0

for i, example in enumerate(dataset):
    if i >= sample_size:
        break
    total_sentences += count_sentences(example['text'])

# Estimate total
avg_sentences_per_example = total_sentences / sample_size
estimated_total = avg_sentences_per_example * len(dataset)

print(f"Sample size: {sample_size} examples")
print(f"Sentences in sample: {total_sentences:,}")
print(f"Average sentences per example: {avg_sentences_per_example:.2f}")
print(f"Total examples in dataset: {len(dataset):,}")
print(f"Estimated total sentences: {estimated_total:,.0f}")
