#! -*- coding: utf-8 -*-

import os

# AMP要使用 tf.keras 
os.environ["TF_KERAS"] = "1"

# bert4keras 自带 重计算
#os.environ["RECOMPUTE"] = "1"

'''
            无重计算   bert4keras重计算    tf_gradients  
bert         8           52                 62
albert       32          n/a                76
'''
import memory_saving_gradients as gc
from tensorflow.python.ops import gradients as tf_gradients
tf_gradients.gradients = gc.gradients_collection


import json, shutil
import numpy as np
from bert4keras.backend import keras, K, tf
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Layer, Dense, Permute
from keras.models import Model
from tqdm import tqdm

# 基本信息
maxlen = 512
epochs = 10
batch_size = 64
learing_rate = 2e-5
model_type = 'albert'

# 百度 阅读理解
# 直接用RoBERTa+Softmax预测首尾
# BASE模型在第一期测试集上能达到0.69的F1，优于官方baseline
# 如果你显存足够，可以换用RoBERTa Large模型，F1可以到0.71
train_data_file1 = '../nlp_model/dureader_robust-data/train.json'  
#eva_data_file = '../nlp_model/dureader_robust-data/dev.json'
#eva_script = 'evaluate/evaluate_dureader.py'

# CMRC2018
#train_data_file2 = '../nlp_model/cmrc2018/cmrc2018_train_dev.json'
#eva_data_file = '../nlp_model/cmrc2018/cmrc2018_trial.json'
train_data_file2 = '../nlp_model/cmrc2018/test_train.json'
eva_data_file = '../nlp_model/cmrc2018/test_dev.json'
eva_script = 'evaluate/evaluate_cmrc2018.py'

'''
bert4keras 支持的 BERT model_type
    'bert': BERT,
    'albert': ALBERT,
    'albert_unshared': ALBERT_Unshared,
    'roberta': BERT,
'''
if model_type=='bert':
    # bert配置
    config_path = '../nlp_model/chinese_bert_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = '../nlp_model/chinese_bert_L-12_H-768_A-12/bert_model.ckpt'
    dict_path = '../nlp_model/chinese_bert_L-12_H-768_A-12/vocab.txt'
elif model_type=='albert':
    # albert配置
    config_path = '../nlp_model/albert_zh_base/albert_config.json'
    checkpoint_path = '../nlp_model/albert_zh_base/model.ckpt-best'
    dict_path = '../nlp_model/albert_zh_base/vocab_chinese.txt'
else:
    print('unknow model type.')
    sys.exit(1)


# 输出目录
output_path = 'outputs/keras_QA'
output_path = '%s/%s_batch%d_max%d_lr%.0e'%(output_path, model_type, batch_size, maxlen, learing_rate)


# 兼容两个数据集的载入
def load_data(filename_list):
    D = []
    for filename in filename_list:
        for d in json.load(open(filename))['data']:
            for pp in d['paragraphs']:
                for qa in pp['qas']:
                    D.append([
                        qa['id'], pp['context'], qa['question'],
                        [a['text'] for a in qa.get('answers', [])]
                    ])
    return D


# 读取数据, 使用两个数据集一起训练
#train_data = load_data([train_data_file1, train_data_file2])
train_data = load_data([train_data_file2])

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            context, question, answers = item[1:]
            token_ids, segment_ids = tokenizer.encode(
                question, context, maxlen=maxlen
            )
            a = np.random.choice(answers)
            a_token_ids = tokenizer.encode(a)[0][1:-1]
            start_index = search(a_token_ids, token_ids)
            if start_index != -1:
                labels = [[start_index], [start_index + len(a_token_ids) - 1]]
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append(labels)
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_labels = sequence_padding(batch_labels)
                    yield [batch_token_ids, batch_segment_ids], batch_labels
                    batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 优化器，使用AMP
opt = Adam(learing_rate)
opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)

# 建立模型，载入权重
class MaskedSoftmax(Layer):
    """在序列长度那一维进行softmax，并mask掉padding部分
    """
    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask, 2)
            inputs = inputs - (1.0 - mask) * 1e12
        return K.softmax(inputs, 1)


model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model=model_type
)


output = Dense(2)(model.output)
output = MaskedSoftmax()(output)
output = Permute((2, 1))(output)

model = Model(model.input, output)
model.summary()


def sparse_categorical_crossentropy(y_true, y_pred):
    # y_true需要重新明确一下shape和dtype
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    y_true = K.one_hot(y_true, K.shape(y_pred)[2])
    # 计算交叉熵
    return K.mean(K.categorical_crossentropy(y_true, y_pred))


def sparse_accuracy(y_true, y_pred):
    # y_true需要重新明确一下shape和dtype
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    # 计算准确率
    y_pred = K.cast(K.argmax(y_pred, axis=2), 'int32')
    return K.mean(K.cast(K.equal(y_true, y_pred), K.floatx()))


model.compile(
    loss=sparse_categorical_crossentropy,
    optimizer=opt,
    metrics=[sparse_accuracy]
)


def extract_answer(question, context, max_a_len=100):
    """抽取答案函数
    """
    max_q_len = 100
    q_token_ids = tokenizer.encode(question, maxlen=max_q_len)[0]
    c_token_ids = tokenizer.encode(
        context, maxlen=maxlen - len(q_token_ids) + 1
    )[0]
    token_ids = q_token_ids + c_token_ids[1:]
    segment_ids = [0] * len(q_token_ids) + [1] * (len(c_token_ids) - 1)
    c_tokens = tokenizer.tokenize(context)[1:-1]
    mapping = tokenizer.rematch(context, c_tokens)
    probas = model.predict([[token_ids], [segment_ids]])[0]
    probas = probas[:, len(q_token_ids):-1]
    start_end, score = None, -1
    for start, p_start in enumerate(probas[0]):
        for end, p_end in enumerate(probas[1]):
            if end >= start and end < start + max_a_len:
                if p_start * p_end > score:
                    start_end = (start, end)
                    score = p_start * p_end
    start, end = start_end
    return context[mapping[start][0]:mapping[end][-1] + 1]


def predict_to_file(infile, out_file):
    """预测结果到文件，方便提交
    """
    fw = open(out_file, 'w', encoding='utf-8')
    R = {}
    for d in tqdm(load_data([infile])):
        a = extract_answer(d[2], d[1])
        R[d[0]] = a
    R = json.dumps(R, ensure_ascii=False, indent=4)
    fw.write(R)
    fw.close()


def evaluate(filename):
    """评测函数（官方提供评测脚本evaluate.py）
    """
    predict_to_file(filename, 'evaluate.pred.json')
    metrics = json.loads(
        os.popen(
            'python3 %s %s %s'
            % (eva_script, filename, 'evaluate.pred.json')
        ).read().strip()
    )
    return metrics


class Evaluator(keras.callbacks.Callback):
    """评估和保存模型
    """
    def __init__(self):
        self.best_val_f1 = 0.
        self.last_best_path = ''

    def on_epoch_end(self, epoch, logs=None):
        metrics = evaluate(eva_data_file)
        if float(metrics['F1']) >= self.best_val_f1:
            self.best_val_f1 = float(metrics['F1'])
            self.save_weights(metrics['F1'])
        metrics['BEST F1'] = self.best_val_f1
        print(metrics)

    def save_weights(self, f1):
        if os.path.exists(self.last_best_path):
            shutil.rmtree(self.last_best_path)
        best_path = '%s_F1_%s'%(output_path, f1)
        os.makedirs(best_path, exist_ok=True)
        model.save_weights('%s/best_model.weights' % best_path)
        self.last_best_path = best_path


if __name__ == '__main__':

    print("maxlen: ", maxlen)
    print("epochs: ", epochs)
    print("batch_size: ", batch_size)
    print("learing_rate: ", learing_rate)
    print("train data: ", len(train_data))

    train_generator = data_generator(train_data, batch_size)
    evaluator = Evaluator()

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

else:
    pass
    '''
    model.load_weights('outputs/albert_batch64_max512_lr2e-05_F1_82.000/best_model.weights')
    corpus = "深度学习（英语：deep learning）是机器学习的分支，是一种以人工神经网络为架构，对资料进行表征学习\
的算法。深度学习是机器学习中一种基于对数据进行表征学习的算法。观测值（例如一幅图像）可以使用多种方式来表示，如\
每个像素强度值的向量，或者更抽象地表示成一系列边、特定形状的区域等。而使用某些特定的表示方法更容易从实例中学习\
任务（例如，人脸识别或面部表情识别）。深度学习的好处是用非监督式或半监督式的特征学习和分层特征提取高效算法\
来替代手工获取特征。"
    ans = extract_answer("什么是深度学习？", corpus)
    print(ans)
    '''