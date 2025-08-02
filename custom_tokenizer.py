from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.model import BPE
from tokenizers.trainers import BpeTrainer
from pathlib import Path

class CustomBPETokenizer:
    def __init__(self, vocab_size=40000):
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"])

    def train(self, files):
        self.tokenizer.train(files, trainer=self.trainer)

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)
    
    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(path)
        print(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path):
        tokenizer_instance = cls.__new__(cls)
        hf_tokenizer = Tokenizer.from_file(path)
        tokenizer_instance.tokenizer = hf_tokenizer
        return tokenizer_instance

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()
    
    def pad_token_id(self):
        return self.tokenizer.token_to_id("[PAD]") 
    
    def sos_token_id(self):
        return self.tokenizer.token_to_id("[SOS]")
    
    def eos_token_id(self):
        return self.tokenizer.token_to_id("[EOS]")
    
    def unk_token_id(self):
        return self.tokenizer.token_to_id("[UNK]")
    
    def token_to_ids(self, tokens):
        for token in tokens:
            yield self.tokenizer.token_to_id(token)
    
    
    
    
    


        

