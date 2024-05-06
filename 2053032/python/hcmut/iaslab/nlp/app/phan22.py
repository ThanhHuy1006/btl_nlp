from nltk import CFG
from nltk.parse.generate import generate
import os
import nltk
nltk.download('punkt')

# Đường dẫn đến thư mục output
output_dir = os.path.join("..", 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Đọc ngữ pháp từ file
def read_grammar(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Sinh câu từ ngữ pháp
def generate_sentences(grammar, max_sentences=10000):
    cfg = CFG.fromstring(grammar)
    sentences = generate(cfg, n=max_sentences, depth=15)  # depth là độ dài tối đa của câu
    return sentences

def save_sentences(sentences, file_path):
    with open(file_path, 'w') as file:
        for sentence in sentences:
            file.write(' '.join(sentence) + '\n')

# Đường dẫn đến file ngữ pháp
grammar_path = os.path.join("..", 'output','grammar.txt')
# Đường dẫn đến file lưu các câu được sinh ra
samples_path = os.path.join("..", 'output','samples.txt')

# Thực hiện sinh câu
grammar = read_grammar(grammar_path)
sentences = generate_sentences(grammar)
save_sentences(sentences, samples_path)

