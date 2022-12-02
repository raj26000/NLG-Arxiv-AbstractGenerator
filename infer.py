import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch


class Inference:
    def __init__(self):
        with open('config.json', 'rb') as f:
            self.config = json.load(f)
        self.config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GPT2LMHeadModel.from_pretrained(self.config['hf_finetuned_checkpoint']).to(self.config['device'])
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.config['hf_finetuned_checkpoint'])

    def generate_abstract(self,
                          title,
                          decoding_strategy='Contrastive Search',
                          num_beams=10,
                          early_stopping=True,
                          no_repeat_ngram_size=3,
                          max_length=1024,
                          top_k=50,
                          top_p=0.95,
                          temperature=0.7,
                          penalty_alpha=0.6):
        prompt = self.config['bos'] + ' ' + title + ' ' + '<|SEP|>'
        prompt_input_ids = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0).to(self.config['device'])
        prompt_attn_mask = torch.tensor(self.tokenizer(prompt)['attention_mask']).unsqueeze(0).to(self.config['device'])
        self.model.eval()
        if decoding_strategy == 'Greedy Search':
            decoded_output = self.model.generate(
                inputs=prompt_input_ids,
                do_sample=False,
                num_beams=1,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_length=max_length,
                attention_mask=prompt_attn_mask
            )
            return self.tokenizer.decode(decoded_output[0], skip_special_tokens=True)

        if decoding_strategy == 'Beam Search':
            decoded_output = self.model.generate(
                inputs=prompt_input_ids,
                do_sample=False,
                num_beams=num_beams,
                early_stopping=early_stopping,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_length=max_length,
                attention_mask=prompt_attn_mask
            )
            return self.tokenizer.decode(decoded_output[0], skip_special_tokens=True)

        elif decoding_strategy == 'Stochastic Sampling':
            decoded_output = self.model.generate(
                inputs=prompt_input_ids,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                max_length=max_length,
                temperature=temperature,
                attention_mask=prompt_attn_mask
            )
            return self.tokenizer.decode(decoded_output[0], skip_special_tokens=True)

        else:
            decoded_output = self.model.generate(
                inputs=prompt_input_ids,
                do_sample=False,
                top_k=top_k,
                max_length=max_length,
                penalty_alpha=penalty_alpha,
                attention_mask=prompt_attn_mask
            )
            return self.tokenizer.decode(decoded_output[0], skip_special_tokens=True)
