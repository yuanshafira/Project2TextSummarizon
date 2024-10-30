---
datasets:
- fajrikoto/id_liputan6
language:
- id
base_model:
- cahya/bert2bert-indonesian-summarization
library_name: transformers
pipeline_tag: summarization
---

# Fine-Tuned BERT2BERT Summarization Model

This model is fine-tuned based on the original [BERT2BERT Indonesian Summarization](https://huggingface.co/cahya/bert2bert-indonesian-summarization) model.

### Fine-Tuned Dataset:
- **Dataset**: [Liputan6_ID](https://huggingface.co/datasets/fajrikoto/id_liputan6)
- **Task**: Summarization

This model was fine-tuned using the [Liputan6_ID](https://huggingface.co/datasets/fajrikoto/id_liputan6) dataset, which contains Indonesian news articles. The model is optimized for summarizing domain-specific texts from the Liputan6 dataset.

## Code Sample

```python
from transformers import BertTokenizer, EncoderDecoderModel

tokenizer = BertTokenizer.from_pretrained("rowjak/bert-indonesian-news-summarization")
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token
model = EncoderDecoderModel.from_pretrained("rowjak/bert-indonesian-news-summarization")

# 
ARTICLE = ""

# generate summary
input_ids = tokenizer.encode(ARTICLE, return_tensors='pt')
summary_ids = model.generate(input_ids,
            max_length=125, 
            num_beams=2,
            repetition_penalty=2.5, 
            length_penalty=1.0, 
            early_stopping=True,
            no_repeat_ngram_size=2,
            use_cache=True)

summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary_text)
```

Output:

```
---
```
