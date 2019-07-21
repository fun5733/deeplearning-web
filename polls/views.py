# -*- coding: utf-8 -*-

from django.shortcuts import render, get_object_or_404
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time
from ipykernel import kernelapp as app

import os
from tensorflow import keras
from django.http import HttpResponse
from django.urls import reverse
from tensorflow.keras.layers import LSTM
from tensorflow.python.keras.layers import recurrent
from tensorflow.python.keras.layers.recurrent import RNN

import numpy as np
from keras.models import load_model, Model
import random
from .forms import UserForm

np.random.seed(5)

def index(request):
    return render(request, 'polls/index.html')

# access convert website, download midi file
def web(s):
    options = Options()
    options.add_argument('--headless')
    options.add_argument("--start-maximized")
    options.add_argument('--window-size=1920x1080')
    options.add_argument("disable-gpu")
    driver = webdriver.Chrome('C:/driv/chromedriver.exe')
    driver.get('https://colinhume.com/music.aspx')
    elem = driver.find_element_by_id("InBox")
    elem.send_keys(s)
    elem = driver.find_element_by_id("ConvBtn")
    elem.click()

    time.sleep(3)

    d = driver.find_element_by_xpath('//*[@id="MainPanel"]/span[5]/a')
    d.click()
    time.sleep(10)
    driver.quit()

def test(request):
    num = 0
    mood = 0
    if request.method == "POST":
        mood = request.POST["mood"]
    else:
        context_dict = {}
        return render(request, 'polls/index.html', context_dict)

    if mood == '1':
        num = 1
    elif mood == '2':
        num = 2
    elif mood == '3':
        num = 3
    elif mood == 0:
        num = -1
    print("num:")
    print(num)
    return HttpResponse(num)

def index1(request):
    s = sampling()
    web(s)
    return HttpResponse(s)
"""
def sampling():
    a = []
    tf.reset_default_graph()
    # 학습에 필요한 설정값들을 지정합니다.
    data_dir = 'data/abc' # abc 데이터로 학습
    batch_size = 50 # Training : 50, Sampling : 1
    seq_length = 50 # Training : 50, Sampling : 1
    hidden_size = 128   # 히든 레이어의 노드 개수
    learning_rate = 0.002
    num_epochs = 2
    num_hidden_layers = 2
    grad_clip = 5   # Gradient Clipping에 사용할 임계값

    # TextLoader를 이용해서 데이터를 불러옵니다.
    data_loader = TextLoader(data_dir, batch_size, seq_length)
    # 학습데이터에 포함된 모든 단어들을 나타내는 변수인 chars와 chars에 id를 부여해 dict 형태로 만든 vocab을 선언합니다.
    chars = data_loader.chars
    vocab = data_loader.vocab
    vocab_size = data_loader.vocab_size # 전체 단어개수

    # 인풋데이터와 타겟데이터, 배치 사이즈를 입력받기 위한 플레이스홀더를 설정합니다.
    input_data = tf.placeholder(tf.int32, shape=[None, None])  # input_data : [batch_size, seq_length])
    target_data = tf.placeholder(tf.int32, shape=[None, None]) # target_data : [batch_size, seq_length])
    state_batch_size = tf.placeholder(tf.int32, shape=[])      # Training : 50, Sampling : 1

    # RNN의 마지막 히든레이어의 출력을 소프트맥스 출력값으로 변환해주기 위한 변수들을 선언합니다.
    # hidden_size -> vocab_size
    softmax_w = tf.Variable(tf.random_normal(shape=[hidden_size, vocab_size]), dtype=tf.float32)
    softmax_b = tf.Variable(tf.random_normal(shape=[vocab_size]), dtype=tf.float32)

    # num_hidden_layers만큼 LSTM cell(히든레이어)를 선언합니다.
    cells = []
    for _ in range(0, num_hidden_layers):
        cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        cells.append(cell)

    # cell을 종합해서 RNN을 정의합니다.
    cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

    # 인풋데이터를 변환하기 위한 Embedding Matrix를 선언합니다.
    # vocab_size -> hidden_size
    embedding = tf.Variable(tf.random_normal(shape=[vocab_size, hidden_size]), dtype=tf.float32)
    inputs = tf.nn.embedding_lookup(embedding, input_data)

    # 초기 state 값을 0으로 초기화합니다.
    initial_state = cell.zero_state(state_batch_size, tf.float32)

    # 학습을 위한 tf.nn.dynamic_rnn을 선언합니다.
    # outputs : [batch_size, seq_length, hidden_size]
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, dtype=tf.float32)
    # ouputs을 [batch_size * seq_length, hidden_size]] 형태로 바꿉니다.
    output = tf.reshape(outputs, [-1, hidden_size])

    # 최종 출력값을 설정합니다.
    # logits : [batch_size * seq_length, vocab_size]
    logits = tf.matmul(output, softmax_w) + softmax_b
    probs = tf.nn.softmax(logits)

    # Cross Entropy 손실 함수를 정의합니다.
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=target_data))

    # 옵티마이저를 선언하고 옵티마이저에 Gradient Clipping을 적용합니다.
    # grad_clip(=5)보다 큰 Gradient를 5로 Clippin합니다.
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.apply_gradients(zip(grads, tvars))


    # tf.train.Saver를 이용해서 모델과 파라미터를 저장합니다.
    SAVER_DIR = "model"
    saver = tf.train.Saver()
    checkpoint_path = os.path.join(SAVER_DIR, "model")
    ckpt = tf.train.get_checkpoint_state(SAVER_DIR)

    log_path = 'pyweb'

    # 세션을 열고 학습을 진행합니다.
    with tf.Session() as sess:
        # 변수들에 초기값을 할당합니다.
        sess.run(tf.global_variables_initializer())
        summary = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())
        for e in range(num_epochs):
            data_loader.reset_batch_pointer()
            # 초기 상태값을 지정합니다.
            state = sess.run(initial_state, feed_dict={state_batch_size : batch_size})
        # 샘플링 시작
        print("샘플링을 시작합니다!")
        num_sampling = 4000  # 생성할 글자(Character)의 개수를 지정합니다.
        prime = u'X: '         # 시작 글자를 'X: '으로 지정합니다.
        sampling_type = 1    # 샘플링 타입을 설정합니다.
        saver.restore(sess, ckpt.model_checkpoint_path)
        state = sess.run(cell.zero_state(1, tf.float32)) # RNN의 최초 state값을 0으로 초기화합니다.

        # Random Sampling을 위한 weighted_pick 함수를 정의합니다.
        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime       # 샘플링 결과를 리턴받을 ret 변수에 첫번째 글자를 할당합니다.
        char = prime[-1]   # Char-RNN의 첫번쨰 인풋을 지정합니다.
        for n in range(num_sampling):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]

            # RNN을 한스텝 실행하고 Softmax 행렬을 리턴으로 받습니다.
            feed_dict = {input_data: x, state_batch_size : 1, initial_state: state}
            [probs_result, state] = sess.run([probs, final_state], feed_dict=feed_dict)

            # 불필요한 차원을 제거합니다.
            # probs_result : (1,65) -> p : (65)
            p = np.squeeze(probs_result)

            # 샘플링 타입에 따라 3가지 종류로 샘플링 합니다.
            # sampling_type : 0 -> 다음 글자를 예측할때 항상 argmax를 사용
            # sampling_type : 1(defualt) -> 다음 글자를 예측할때 항상 random sampling을 사용
            # sampling_type : 2 -> 다음 글자를 예측할때 이전 글자가 ' '(공백)이면 random sampling, 그렇지 않을 경우 argmax를 사용
            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred     # 샘플링 결과에 현재 스텝에서 예측한 글자를 추가합니다. (예를들어 pred=L일 경우, ret = HEL -> HELL)
            char = pred     # 예측한 글자를 다음 RNN의 인풋으로 사용합니다.

        print("샘플링 결과:")
        # print(ret)
        a.append(ret)


    #aat = a.split('X:')
    #s = "X:" + aat[0]
    #print(s)
    #return HttpResponse("%s" %(a))
    #return HttpResponse("%S", s)
    # return HttpResponse("Success!")
    print("a의 타입은")
    print(type(a))
    return a
"""
def open_seq(code_list):
    data = []
    learn_data = []
    data_ls = []
    data_num = []
    for i in range(len(code_list)):
        for j in range(len(code_list[i])):
            data.append(code_list[i][j])
    for i in data:
        if not i in data_ls:
            data_ls.append(i)
    data_ls.append(':|')
    data_ls.append('|:')
    for j in range(0, len(data_ls)):
        data_num.append(j)
    data2num = dict(zip(data_ls, data_num))
    num2data = dict(zip(data_num, data_ls))
    for i in range(len(data)):
        tmp = ''
        if data[i] is '|' and data[i-1] is ':':
            tmp = ':|'
            learn_data.pop()
        elif data[i] is ':' and data[i-1] is '|':
            tmp = '|:'
            learn_data.pop()
        else:
            tmp = data[i]
        learn_data.append(tmp)
    return learn_data, data2num, num2data

def open_file(filename):
    f = open(filename, 'r', encoding='utf-16')
    M = []
    L = []
    K = []
    Q = []
    X = []
    tmp = ''
    count = 0
    while True:
        line = f.readline()
        if not line: break
        if line[0] is 'X':
            count = 0
            X.append(tmp)
            tmp = ''
        if line[0] is 'M':
            M.append(line[2:])
        if line[0] is 'L':
            L.append(line[2:])
        if line[0] is 'K':
            K.append(line[2:])
            count = count + 1
            continue
        if line[0] is 'Q':
            Q.append(line[2:])
        if len(Q) < len(K):
            Q.append('no')
        if len(L) < len(K):
            L.append('no')
        if len(M) < len(K):
            M.append('no')
        if count is 1:
            tmp = tmp + line
    f.close()
    X.append(tmp)
    del X[0]
    return M, L, K, Q, X

def sampling():
    n_steps = 4  # step
    n_inputs = 1  # 특성수
    model = load_model("model.h5")

    rhythm, code_len, chords, quick, X_code = open_file("happy.txt")
    seq, code2idx, idx2code = open_seq(X_code)
    max_idx_value = len(code2idx) - 1
    pred_count = 50  # 최대 예측 개수 정의

    # 곡 전체 예측

    seq_in = ['G', 'F', 'E', 'D']
    seq_out = seq_in
    seq_in = [code2idx[it] / float(max_idx_value) for it in seq_in]  # 코드를 인덱스값으로 변환

    for i in range(pred_count):
        sample_in = np.array(seq_in)
        sample_in = np.reshape(sample_in, (1, n_steps, n_inputs))  # 샘플 수, 타입스텝 수, 속성 수
        pred_out = model.predict(sample_in)
        idx = np.argmax(pred_out)
        seq_out.append(idx2code[idx])
        seq_in.append(idx / float(max_idx_value))
        seq_in.pop(0)

    model.reset_states()

    m_result = ''.join(random.sample(rhythm, 1))
    l_result = ''.join(random.sample(code_len, 1))
    k_result = ''.join(random.sample(chords, 1))

    print("full song prediction : ")
    result = "X: 1\nT: sample\nM: " + m_result + "\nL: " + l_result + "\nK: " + k_result
    print(result)
    print(''.join(seq_out))
    return result

#


