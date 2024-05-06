import random
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Your data
# Your data
intents={
     "intents":[
        {
            "tag": "chao_hoi",
            "patterns": [
                "Xin chào",
                "Chào bạn",
                "Có ai ở đó không?",
                "Chào bạn, bạn khỏe không?",
                "Xin chào bạn, bạn đang làm gì?"
            ],
            "responses": [
                "Xin chào! Rất vui được gặp bạn.",
                "Chào bạn, cảm ơn bạn đã ghé thăm.",
                "Xin chào, bạn cần giúp gì không?",
                "Chào bạn, tôi sẵn lòng hỗ trợ bạn."
            ]
        },
        {
            "tag": "tam_biet",
            "patterns": [
                "Tạm biệt",
                "Hẹn gặp lại sau nhé",
                "Chúc bạn một ngày tốt lành",
                "Hẹn gặp lại! Hãy quay lại sớm nhé."
            ],
            "responses": [
                "Hẹn gặp lại bạn, cảm ơn bạn đã ghé thăm.",
                "Chúc bạn một ngày tốt lành!",
                "Tạm biệt! Hãy quay lại sớm.",
                "Hẹn gặp lại bạn sau nhé, chúc bạn mọi điều tốt lành."
            ]
        },
        {
            "tag": "cam_on",
            "patterns": [
                "Cảm ơn",
                "Rất biết ơn",
                "Đã giúp được gì không?",
                "Cảm ơn bạn rất nhiều!"
            ],
            "responses": [
                "Rất vui được giúp đỡ bạn!",
                "Không có gì, luôn sẵn lòng giúp đỡ bạn.",
                "Đừng ngần ngại, tôi luôn ở đây để hỗ trợ bạn.",
                "Cảm ơn bạn, hãy để tôi biết nếu bạn cần thêm sự giúp đỡ nào khác."
            ]
        },
        {
            "tag": "dich_vu_khach_hang",
            "patterns": [
                "Làm thế nào để liên hệ với bộ phận dịch vụ khách hàng?",
                "Có dịch vụ hỗ trợ khách hàng không?",
                "Tôi cần hỗ trợ, làm thế nào?"
            ],
            "responses": [
                "Bạn có thể gọi số hotline của chúng tôi để liên hệ với bộ phận dịch vụ khách hàng.",
                "Dịch vụ khách hàng của chúng tôi luôn sẵn lòng hỗ trợ bạn qua điện thoại hoặc email.",
                "Hãy gửi email cho chúng tôi để được hỗ trợ nhanh nhất."
            ]
        },
        {
            "tag": "dich_vu_giao_hang",
            "patterns": [
                "Cửa hàng có dịch vụ giao hàng không?",
                "Làm thế nào để đặt hàng online?",
                "Tôi muốn đặt hàng, làm thế nào?"
            ],
            "responses": [
                "Chúng tôi có dịch vụ giao hàng tận nơi, bạn có thể đặt hàng online hoặc qua điện thoại.",
                "Để đặt hàng online, bạn có thể truy cập website của chúng tôi và làm theo hướng dẫn.",
                "Hãy gọi cho chúng tôi để đặt hàng và chúng tôi sẽ giao hàng đến địa chỉ của bạn."
            ]
        },
        {
    "intents": [
        {
            "tag": "cau_hinh_laptop",
            "patterns": [
                "Tôi muốn biết cấu hình của laptop",
                "Có thể cho tôi biết cấu hình của một chiếc laptop không?",
                "Tôi cần thông tin về cấu hình laptop."
            ],
            "responses": [
                "Dĩ nhiên! Dưới đây là một số thông tin cơ bản về cấu hình của laptop:\n- CPU: Bạn muốn biết về CPU nào? (ví dụ: Intel Core i5, AMD Ryzen 7)\n- RAM: Bao nhiêu GB RAM bạn quan tâm?\n- Ổ cứng: SSD hay HDD? Bao nhiêu dung lượng?\n- Card đồ họa: Bạn cần thông tin về card đồ họa nào? (ví dụ: NVIDIA GeForce GTX 1650, Integrated Intel UHD Graphics)"
            ]
        },
        {
            "tag": "gia_laptop",
            "patterns": [
                "Tôi muốn biết giá của laptop",
                "Có thể cho tôi biết giá của một chiếc laptop không?",
                "Tôi cần thông tin về giá của laptop."
            ],
            "responses": [
                "Tùy thuộc vào thương hiệu, dòng sản phẩm và cấu hình cụ thể, giá của laptop có thể dao động từ mức giá thấp đến cao. Bạn có thể cung cấp thông tin chi tiết hơn để chúng tôi có thể tìm giúp bạn."
            ]
        },
        {
            "tag": "thong_tin_cu_the_laptop",
            "patterns": [
                "Tôi muốn biết thông tin chi tiết của một chiếc laptop",
                "Có thể cho tôi biết thông tin chi tiết của một chiếc laptop không?",
                "Tôi cần biết cụ thể về một chiếc laptop."
            ],
            "responses": [
                "Tất nhiên! Bạn muốn biết thông tin chi tiết của laptop nào? Xin vui lòng cung cấp thêm thông tin về thương hiệu, dòng sản phẩm và cấu hình mong muốn."
            ]
        }
    ]
}

    ]
}

# Prepare the data for training
all_words = []
tags = []
xy = []

# Your preprocessing here
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = word_tokenize(pattern)
        all_words.extend(w)
        xy.append((w, intent['tag']))

ignore_words = ['?', '!', '.']
all_words = [PorterStemmer().stem(w.lower()) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tag['tag'] for tag in intents['intents']))

training = []
output_empty = [0] * len(tags)

for (pattern_sentence, tag) in xy:
    bag = [1 if w in pattern_sentence else 0 for w in all_words]
    output_row = list(output_empty)
    output_row[tags.index(tag)] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Create model
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5)

# Function to get a response
def get_response(sentence):
    sentence_words = word_tokenize(sentence)
    sentence_words = [PorterStemmer().stem(word.lower()) for word in sentence_words]
    bag = [1 if w in sentence_words else 0 for w in all_words]
    res = model.predict(np.array([bag]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    if results:
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append((tags[r[0]], r[1]))
        tag = return_list[0][0]
        for tg in intents['intents']:
            if tg['tag'] == tag:
                return random.choice(tg['responses'])
    return "I don't understand what you said."

# Main loop to process input
while True:
    message = input("You: ")
    if message.lower() == "quit":
        break
    response = get_response(message)
    print("Bot:", response)
