import nltk
import os
nltk.download('punkt')

# Đường dẫn đến thư mục input và output
input_dir = os.path.join("..", 'input')
output_dir = os.path.join("..", 'output')

# Đọc câu từ file input
input_file = os.path.join(input_dir, 'sentences.txt')
with open(input_file, 'r') as file:
    sentences = file.readlines()

# Đường dẫn đến file ngữ pháp
grammar_file = os.path.join(output_dir, 'grammar.txt')
with open(grammar_file, 'r') as file:
    grammar = file.read()

# Khởi tạo parser từ ngữ pháp đã định nghĩa
parser = nltk.ChartParser(nltk.CFG.fromstring(grammar))

# Phân tích cú pháp cho từng câu và ghi kết quả vào file output
output_file = os.path.join(output_dir, 'parse-results.txt')
with open(output_file, 'w') as file:
    for sentence in sentences:
        # Loại bỏ ký tự xuống dòng và khoảng trắng thừa ở hai đầu câu
        sentence = sentence.strip()
        if sentence:
            try:
                # Phân tích cú pháp cho câu
                parsed_tree = next(parser.parse(sentence.split()))
                file.write(str(parsed_tree) + '\n')
            except Exception as e:
                # Nếu câu không hợp lệ, xuất ra cây rỗng
                file.write("()\n")
                print(f"Error parsing sentence: {sentence} - {e}")
