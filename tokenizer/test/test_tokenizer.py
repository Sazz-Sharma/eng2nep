import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


from custom_tokenizer import CustomBPETokenizer

tokenizer = CustomBPETokenizer(vocab_size=1000)

corpus_file = project_root/'dummy_corpus.txt'
tokenizer.train([str(corpus_file)])
tokenizer.save('temp/test_tokenizer.json')


