# -*- coding: UTF-8 -*-
# 訓練一個模型，用來辨識電影的評論是正向的還是負向的

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import sys

def main():
    print('-------------- 檔案列表 --------------')
    print(sys.argv)
    print('-------------- START --------------')
    # 載入內含 50,000 部電影評論的資料集
    # num_words=10000 : 保留資料集中最常出現的 10,000 個單字
    # 註：因為這個檔案內含在 TF 裡，所以直接呼叫即可
    # 這邊要注意如果 numpy 的版本大於 1.16.2 的話會出錯 ValueError: Object arrays cannot be loaded when allow_pickle=False
    # pip uninstall numpy  
    # pip install numpy==1.16.2 --user
    # ref: https://qiita.com/INM/items/a790676455d103c05ed2  
    imdb = keras.datasets.imdb
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    # Explore the data
    # 會得到 Training entries: 25000, labels: 25000
    print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
    # 會得到 [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2025, 19, 14, 22, 4, 1920, 4613, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 1247, 4, 22, 17, 515, 17, 12, 16, 626, 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 5244, 16, 480, 66, 3785, 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 5952, 15, 256, 4, 2, 7, 3766, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 2, 1029, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2071, 56, 26, 141, 6, 194, 7486, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 5535, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113, 103, 32, 15, 16, 5345, 19, 178, 32]
    print(train_data[0])
    # 會得到 (218, 189)
    print(len(train_data[0]), len(train_data[1]))

    # Convert the integers back to words
    # 把 integer 轉回文字

    # A dictionary mapping words to an integer index
    word_index = imdb.get_word_index()

    # The first indices are reserved
    word_index = {k:(v+3) for k,v in word_index.items()} 
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    def decode_review(text):
      return ' '.join([reverse_word_index.get(i, '?') for i in text])

    # 會得到解碼字串
    print(decode_review(train_data[0]))


    # Prepare the data
    # 我們需要把評論 ( 整數陣列 )轉換成 tensor，才有辦法接下來的動作，一般有兩種方法：
    # 1. 把 array 轉乘 0 和 1 的向量，表示單字有沒有出現，又稱為 one-hot encoding，然後再 layer 作處理
    #    這種做法要用到很多記憶體，需要 num_words * num_reviews 的矩陣
    # 2. 把所有陣列標準化長度(後面補 0)，這樣 tensor 的大小就變成 max_length * num_reviews，處理完再放到 layer
    #    這邊我們用第二個方法
    # 為了採取第二個方法，我們需要用到 pad_sequences 來標準化長度
    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=word_index["<PAD>"],
                                                            padding='post',
                                                            maxlen=256)

    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                           value=word_index["<PAD>"],
                                                           padding='post',
                                                           maxlen=256)
    # 會得到 (256, 256)
    print(len(train_data[0]), len(train_data[1]))
    # 在檢查內容，確定後面補 0
    print(train_data[0])

    # Build the model
    # 模型是用一層又一層的 layer 疊起來的，所以在做 layer 的時候，要問自己兩個問題：
    # 1. model 要用幾個 layer 
    # 2. 每個 layer 使用多少個隱藏單位？


    # 在這個例子中，input 是由單字索引組成，label 不是 0 就是 1
    # input shape is the vocabulary count used for the movie reviews (10,000 words)
    vocab_size = 10000

    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
    model.summary()

    # 每層的解說：
    # Embedding : 用整數編碼的字典，找出單子索引的向量，輸出 (batch, sequence, embedding), 
    # GlobalAveragePooling1D : 藉由對序列維度求平均值，回傳一個固定長度的向量，讓 model 以最簡單方式處理不同長度的 inputs
    # Dense : 這個固定長度的向量，pipe 進有 16 個隱藏單位的第三層 (Dence) 
    # Dense : 找到 0 或 1 的可能性
    # Hidden units 到底是什麼？
    # 就是介於輸入與輸出之間有幾層，這個例子有兩層

    # binary_crossentropy 比 MSE 還更適合在訓練有關可能性的題目上，因為他測量可能性分布的距離

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc'])

    # Create a validation set
    # 因為 test data 是最後用來測試的，所以把 train data 切一點出來當作驗證
    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]

    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]

    # Train the model
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=10,
                        batch_size=512,
                        validation_data=(x_val, y_val),
                        verbose=1)

    # Evaluate the model
    # 用 test_data, test_labels 評估 model 的績效
    results = model.evaluate(test_data, test_labels)
    print('用 test_data, test_labels 評估 model 的績效')
    print(results)