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
        """
        Method to generate the output sequence using the finetuned model for test samples.
        :param title: Title of an arXiv paper with cs.CL category
        :param decoding_strategy: Provide one among - Greedy Search, Beam Search, Stochastic Sampling, Contrastive Search
        :param num_beams: number of beams for beam search, has to be > 1 to perform beam search
        :param early_stopping: Stopping beam search when any beam hits <|endoftext|> token.
        :param no_repeat_ngram_size: Order of n-gram to avoid repetitions in outputs
        :param max_length: Max length of output sequence (including input title prompt)
        :param top_k: choose top-k most probable words from decoder output before sampling or contrastive search.
        :param top_p: to choose the smallest subset of words from vocab such that sum of probabilities hits p, before sampling.
        :param temperature: to scale logits by a factor before applying softmax to the vocab subset during sampling.
        :param penalty_alpha: Model degeneracy penalty weight 'alpha' in contrastive search objective function. To penalize repetitive tokens.
        :return: Decoded output sequence, from which abstract is later separated.
        """
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

        elif decoding_strategy == 'Beam Search':
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
