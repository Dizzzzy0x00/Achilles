# filepath: f:\AFL\javaVulnerability\Achilles\Model.py
import pandas as pd
import sys
import os
# Disables printing "Using XXX backend", because it pisses me off.
# stderr = sys.stderr
# sys.stderr = open(os.devnull, 'w')
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from constants import *

import logging

# 配置日志记录
logging.basicConfig(
    filename='training.log',  # 日志文件名
    level=logging.INFO,       # 日志级别
    format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
)


class AchillesModel:
    @staticmethod
    def RNN():
        inputs = Input(name='inputs', shape=[MAX_LEN])
        layer = Embedding(MAX_WORDS, 50)(inputs)
        layer = LSTM(64)(layer)
        layer = Dense(256, name='FC1')(layer)
        layer = Activation(ACTIVATION_FUNCT)(layer)
        layer = Dropout(DROPOUT_RATE)(layer)
        layer = Dense(1, name='out_layer')(layer)
        layer = Activation('sigmoid')(layer)
        model = Model(inputs=inputs, outputs=layer)
        return model

    @staticmethod
    def train(df, write_h5_to):
        if isinstance(df, str):
            df = pd.read_csv(df)
        X = df.input
        Y = df.label
        le = LabelEncoder()
        Y = le.fit_transform(Y)
        Y = Y.reshape(-1, 1)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE)

        tok = Tokenizer(num_words=MAX_WORDS)
        tok.fit_on_texts(X_train)
        sequences = tok.texts_to_sequences(X_train)
        sequences_matrix = sequence.pad_sequences(sequences, maxlen=MAX_LEN)

        model = AchillesModel.RNN()
        model.summary()
        model.compile(loss=LOSS_FUNCT, optimizer=RMSprop(learning_rate=0.001), metrics=['accuracy'])
        hist = model.fit(sequences_matrix, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                         validation_split=VALIDATION_SPLIT, callbacks=[EarlyStopping(monitor='val_loss', min_delta=MIN_DELTA)])
        test_sequences = tok.texts_to_sequences(X_test)
        test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=MAX_LEN)
        accr = model.evaluate(test_sequences_matrix, Y_test)
        model.save(write_h5_to, overwrite=True)
        print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}\n'.format(accr[0], accr[1]))


    @staticmethod
    def continue_training(df, model_path, epochs=20):
        """
        加载已保存的模型并继续训练。
        :param df: 包含 'input' 和 'label' 列的 pandas DataFrame。
        :param model_path: 已保存模型的路径 (.h5 文件)。
        :param epochs: 继续训练的轮数。
        :return: None
        """
        from keras.models import load_model
        from keras.callbacks import EarlyStopping, LambdaCallback
        from sklearn.model_selection import train_test_split

        # 检查输入数据格式
        if 'input' not in df.columns or 'label' not in df.columns:
            msg = "DataFrame must contain 'input' and 'label' columns."
            print(f"\x1b[31m{msg}\x1b[m")
            logging.error(msg)
            return

        # 加载模型
        if not os.path.isfile(model_path):
            msg = f"Model file not found: {model_path}"
            print(f"\x1b[31m{msg}\x1b[m")
            logging.error(msg)
            return
        model = load_model(model_path)
        msg = f"Loaded model from {model_path}"
        print(msg)
        logging.info(msg)

        # 重新编译模型
        model.compile(loss=LOSS_FUNCT, optimizer=RMSprop(lr=0.001), metrics=['accuracy'])
        msg = f"Recompiled model with loss={LOSS_FUNCT} and metrics=['accuracy']"
        print(msg)
        logging.info(msg)

        # 准备数据
        X = df['input']
        Y = df['label']

        # 划分训练集和测试集
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=42)

        # 预处理数据
        tok = Tokenizer(num_words=MAX_WORDS)
        tok.fit_on_texts(X_train)
        train_sequences = tok.texts_to_sequences(X_train)
        train_sequences_matrix = sequence.pad_sequences(train_sequences, maxlen=MAX_LEN)

        test_sequences = tok.texts_to_sequences(X_test)
        test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=MAX_LEN)

        # 定义每轮训练的回调函数
        def on_epoch_end(epoch, logs):
            msg = f"Epoch {epoch + 1}/{epochs} - Loss: {logs['loss']:.4f}, Accuracy: {logs['acc']:.4f}, Val_Loss: {logs['val_loss']:.4f}, Val_Accuracy: {logs['val_acc']:.4f}"
            print(msg)
            logging.info(msg)

        epoch_callback = LambdaCallback(on_epoch_end=on_epoch_end)
        # 继续训练
        msg = f"Continuing training {model_path} for {epochs} epochs..."
        print(msg)
        logging.info(msg)
        hist = model.fit(
            train_sequences_matrix, Y_train,
            batch_size=BATCH_SIZE,
            epochs=epochs,
            validation_split=VALIDATION_SPLIT,
            callbacks=[
                EarlyStopping(monitor='val_loss', min_delta=MIN_DELTA, patience=5),
                epoch_callback
            ]
        )

        # 评估模型
        accr = model.evaluate(test_sequences_matrix, Y_test)
        msg = 'Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}\n'.format(accr[0], accr[1])
        print(msg)
        logging.info(msg)

        # 保存更新后的模型
        model.save(model_path, overwrite=True)
        msg = f"Updated model saved to {model_path}"
        print(msg)
        logging.info(msg)

    @staticmethod
    def evaluate(df, model_path):
        """
        对给定的 DataFrame 和模型进行评估。
        :param df: 包含 'input' 和 'label' 列的 pandas DataFrame。
        :param model_path: 已训练模型的路径 (.h5 文件)。
        :return: None
        """
        from keras.models import load_model
        from sklearn.metrics import accuracy_score, f1_score

        # 检查输入数据格式
        if 'input' not in df.columns or 'label' not in df.columns:
            print("\x1b[31mDataFrame must contain 'input' and 'label' columns.\x1b[m")
            return

        # 加载模型
        if not os.path.isfile(model_path):
            print(f"\x1b[31mModel file not found: {model_path}\x1b[m")
            return
        model = load_model(model_path)

        # 准备数据
        X = df['input']
        Y = df['label']

        # 划分 10% 的数据用于测试
        _, X_test, _, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=42)

        # 预处理数据
        tok = Tokenizer(num_words=MAX_WORDS)
        tok.fit_on_texts(X)
        test_sequences = tok.texts_to_sequences(X_test)
        test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=MAX_LEN)

        # 预测
        predictions = model.predict(test_sequences_matrix)
        predictions = (predictions > 0.5).astype(int).flatten()

        # 计算评估指标
        accuracy = accuracy_score(Y_test, predictions)
        f1 = f1_score(Y_test, predictions)

        # 输出评估结果
        print(f"Model: {model_path}")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  F1 Score: {f1:.3f}")