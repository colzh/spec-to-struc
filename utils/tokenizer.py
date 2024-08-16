import json
from tqdm import tqdm
import torch

class Tokenizer():
    def __init__(self):
        self.vocab = {}
        self.cls_token = '<s>'
        self.sep_token = '</s>'
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'

    def fit(self, smiles):
        # Initialize the vocabulary with special tokens
        self.vocab = {self.cls_token: 0, self.sep_token: 1, self.pad_token: 2, self.unk_token: 3}

        # Store IDs for special tokens
        self.cls_token_id = self.vocab[self.cls_token]
        self.sep_token_id = self.vocab[self.sep_token]
        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]
        
        # Fill the vocabulary with unique characters from the smiles
        idx = len(self.vocab)
        for s in tqdm(smiles):
            for char in s:
                if char not in self.vocab:
                    self.vocab[char] = idx
                    idx += 1

    def save_vocab(self, vocab_path):
        with open(vocab_path, 'w') as f:
            json.dump(self.vocab, f)

    def load_vocab(self, vocab_path):
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        
        # Store IDs for special tokens
        self.cls_token_id = self.vocab[self.cls_token]
        self.sep_token_id = self.vocab[self.sep_token]
        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]

        return self

    def tokenize(self, smiles, max_length):
        # Get token IDs for each character in the SMILES
        encoded = [self.cls_token_id] + [self.vocab.get(char, self.unk_token_id) for char in smiles] + [self.sep_token_id]
        
        # Truncate if necessary
        if len(encoded) > max_length:
            encoded = encoded[:max_length]

        # Pad if necessary
        while len(encoded) < max_length:
            encoded.append(self.pad_token_id)

        return torch.tensor(encoded)

    def decode(self, toks):

        # Convert to a list if necessary
        if not isinstance(toks, list):
            toks = toks.tolist()

        # Get the reverse of the vocabulary
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Decode the token ids using the reverse vocabulary
        decoded = [reverse_vocab.get(token_id) for token_id in toks]

        # Skip the special tokens
        special_tokens = [self.cls_token, self.sep_token, self.pad_token, self.unk_token]
        decoded = [elem for elem in decoded if elem not in special_tokens]
        
        # Join the characters to get the SMILES/SELFIES string
        result = ''.join(decoded)

        return result
