from transformers import FlaubertTokenizer, FlaubertModel, FlaubertConfig, FlaubertForTokenClassification
import torch

tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_base_cased')
config = FlaubertConfig.from_pretrained('flaubert/flaubert_base_cased')
model = FlaubertModel.from_pretrained('flaubert/flaubert_base_cased',
                                      config=config)

inputs = tokenizer("Bonjour, je m'appelle Maxime et ceci est un test. Gardez tous votre calme.",
                    return_tensors="pt")
print(inputs.input_ids.shape)
outputs = model(**inputs)


pred_logits = outputs.last_hidden_state
# predictions = torch.argmax(outputs, dim=2)
print(pred_logits.shape)
