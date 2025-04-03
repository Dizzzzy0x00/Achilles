import javalang
import re
import random
import string
import numpy as np
import csv
from sklearn.utils.extmath import softmax
from datetime import datetime
from model import *


# JavaClass: receives a `path` of a Java file, extracts the code,
# converts the code from Allman to K&R, extracts the methods,
# and creates a list of JavaMethod objects.
class JavaClass:
    def __init__(self, path):
        self.src = JavaClass._extract_code(path)
        self.src = JavaClass._allman_to_knr(self.src)
        self.methods = JavaClass.chunker(self.src)
        self.method_names = [method.name for method in self.methods]

    # __iter__: iterate through the methods in the JavaClass.
    def __iter__(self):
        return iter(self.methods)

    # _extract_code: receives a `path` to a Java file, removes all comments,
    # and returns the contents of the file sans comments.
    @staticmethod
    def _extract_code(path):
        with open(path, 'r') as content_file:
            contents = content_file.read()
            contents = re.sub(re.compile("/\*.*?\*/", re.DOTALL), "", contents)
            contents = re.sub(re.compile("//.*?\n"),  "", contents)
            return contents

    # tokens: A getter that returns a 1-dimensional space-delimited
    # string of tokens for the whole source file.
    def tokens(self):
        tokens = javalang.tokenizer.tokenize(self.src)
        return [" ".join(token.value for token in tokens)][0]

    # find_occurences: Returns a list of all occurrences of
    # a character `ch` in a string `s`.
    @staticmethod
    def find_occurrences(s, ch):
        return [i for i, letter in enumerate(s) if letter == ch]

    # _allman_to_knr: Converts a string `contents` from the style of
    # allman to K&R. This is required for `chunker` to work correctly.
    @staticmethod
    def _allman_to_knr(contents):
        s, contents = [], contents.split("\n")
        line = 0
        while line < len(contents):
            if contents[line].strip() == "{":
                s[-1] = s[-1].rstrip() + " {"
            else:
                s.append(contents[line])
            line += 1
        return "\n".join(s)

    # chunker: Extracts the methods from `contents` and returns
    # a list of `JavaMethod` objects.
    @staticmethod
    def chunker(contents):
        r_brace = JavaClass.find_occurrences(contents, "}")
        l_brace = JavaClass.find_occurrences(contents, "{")
        tokens = javalang.tokenizer.tokenize(contents)
        guide, chunks = "", []
        _blocks = ["enum", "finally", "catch", "do", "else", "for",
                   "if", "try", "while", "switch", "synchronized"]

        for token in tokens:
            if token.value in ["{", "}"]:
                guide += token.value

        while len(guide) > 0:
            i = guide.find("}")
            l, r = l_brace[i - 1], r_brace[0]
            l_brace.remove(l)
            r_brace.remove(r)

            ln = contents[0:l].rfind("\n")
            chunk = contents[ln:r + 1]
            if len(chunk.split()) > 1:
                if chunk.split()[0] in ["public", "private", "protected"] and "class" not in chunk.split()[1]:
                    chunks.append(JavaMethod(chunk))
            guide = guide.replace("{}", "", 1)
        return chunks


# JavaMethod: receives a `chunk`, which is the method string.
class JavaMethod:
    def __init__(self, chunk):
        self.method = chunk
        self.name = chunk[:chunk.find("(")].split()[-1]

    # tokens: a getter that returns a 1-dimensional space-delimited
    # string of tokens for the method.
    def tokens(self):
        tokens = javalang.tokenizer.tokenize(self.method)
        return [" ".join(token.value for token in tokens)][0]

    # __str__: String representation for a method.
    def __str__(self):
        return self.method

    # __iter__: Iterator for the tokens of each method.
    def __iter__(self):
        tokens = javalang.tokenizer.tokenize(self.method)
        return iter([tok.value for tok in tokens])


# CWE4J: receives a path `root` to a folder containing labeled Java
# classes with known vulnerabilities. Each Java file in the `root`
# directory is then added to a dictionary `data` of the form
# {vulnerability: [path_1, path_2, ..., path_n]}
class CWE4J:
    def __init__(self, root):
        self.data = {}
        self.root = root
        for directory in os.listdir(root):
            self.add(directory)

    # add: if the vulnerability already has an entry in `self.data`,
    # the filepath will be appended to the value list. Otherwise,
    # the name of the vulnerability and a list containing the path to
    # the vulnerability are created as the key and value, respectively.
    def add(self, filepath):
        vuln_name = filepath.split("__")[0]
        if vuln_name in self.data.keys():
            self.data[vuln_name].append(self.root + "/" + filepath)
        else:
            self.data[vuln_name] = [self.root + "/" + filepath]

    # __iter__: iterator for the keys of `self.data`.
    def __iter__(self):
        return iter(self.data.keys())

    # __getitem__: allows a CWE4J object to be indexed by key.
    def __getitem__(self, item):
        return self.data[item]

    # __len__: returns the number of keys in `self.data`.
    def __len__(self):
        return len(self.data.keys())


# Javalect: contains functions that are responsible for
# Achilles' core functionality.
class Javalect:
    # train_models: trains several models from appropriately names
    # Java files from a `root` directory. The `threshold` will tell
    # Achilles to ignore any vulnerability categories that contain
    # less than a given number of examples to train on.
    @staticmethod
    def train_models(root, threshold=1000):
        cwe4j = CWE4J(root)
        for cwe in cwe4j:
            if len(cwe4j[cwe]) >= threshold:
                Javalect._train_model(str(cwe), cwe4j[cwe])

    @staticmethod
    def gen_data(root, threshold=1000):
        cwe4j = CWE4J(root)
        print("cwe4j len", len(cwe4j))
        for cwe in cwe4j:
            print("test cwe data of ", cwe)
            print("test cwe data len", len(cwe4j[cwe]))
            if len(cwe4j[cwe]) >= threshold:
                Javalect._generate_dataset(str(cwe), cwe4j[cwe])


    # _train_model: A helper method for `train_models`, that examines
    # the method name of each method in the given training files.
    # If a method name contains the word "bad", it is labeled with a
    # 1; if it contains "good", it is labeled with a 0. The labeled
    # data is returned in a pandas dataframe of the form
    # [tokenized java method, binary polarity bit (0/1)].
    @staticmethod
    def _train_model(cwe_name, cwe_paths):
        df = [["input", "label"]]
        for path in cwe_paths:
            try:
                j = JavaClass(path)
                for method in j.methods:
                    focus = method.tokens().split("(", 1)
                    if "good" in focus[0]:
                        rand = ''.join(random.choices(string.ascii_uppercase + string.digits, k=7))
                        temp = focus[0].replace("good", rand) + "(" + focus[1]
                        df.append([temp, "0"])
                    elif "bad" in focus[0]:
                        rand = ''.join(random.choices(string.ascii_uppercase + string.digits, k=7))
                        temp = focus[0].replace("bad", rand) + "(" + focus[1]
                        df.append([temp, "1"])
            except:
                pass
        dataframe = pd.DataFrame(df[1:], columns=df[0])
        AchillesModel.train(dataframe, os.path.realpath(__file__)[:-11] + "/data/java/checkpoints/" + cwe_name + ".h5")
        with open(os.path.realpath(__file__)[:-11] + '/data/java/vocab.csv', 'a') as fd:
            writer = csv.writer(fd)
            writer.writerows(df[1:])

    

    @staticmethod
    def _generate_dataset(cwe_name, cwe_paths):
        """
        根据给定的 CWE 名称和文件路径列表生成数据集。
        :param cwe_name: CWE 名称（字符串）。
        :param cwe_paths: 包含 Java 文件路径的列表。
        :return: 一个 pandas DataFrame，包含 'input' 和 'label' 列。
        """
        print("gen dataset of ", cwe_name)
        df = [["input", "label"]]
        for path in cwe_paths:
            try:
                j = JavaClass(path)
                for method in j.methods:
                    focus = method.tokens().split("(", 1)
                    if "good" in focus[0]:
                        rand = ''.join(random.choices(string.ascii_uppercase + string.digits, k=7))
                        temp = focus[0].replace("good", rand) + "(" + focus[1]
                        df.append([temp, "0"])
                    elif "bad" in focus[0]:
                        rand = ''.join(random.choices(string.ascii_uppercase + string.digits, k=7))
                        temp = focus[0].replace("bad", rand) + "(" + focus[1]
                        df.append([temp, "1"])
            except Exception as e:
                print(f"Error processing file {path}: {e}")
                pass

        # 转换为 pandas DataFrame
        dataframe = pd.DataFrame(df[1:], columns=df[0])

        # 确保目标目录存在
        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "java", "dataset")
        os.makedirs(output_dir, exist_ok=True)

        # 保存到指定目录
        output_path = os.path.join(output_dir, f"{cwe_name}.csv")
        dataframe.to_csv(output_path, index=False)
        print(f"Dataset saved to: {output_path}")

        return
    
    # _embed: given a Keras tokenizer `tok`, embed a vectorized
    # string to a higher dimension, of size `MAX_LEN`.
    @staticmethod
    def _embed(tok, method):
        sequences = tok.texts_to_sequences([method])
        sequences_matrix = sequence.pad_sequences(sequences, maxlen=MAX_LEN)
        return sequences_matrix

    # analyze: creates a Keras tokenizer with from a vocabulary
    # csv, loads all vulnerability models into memory, then
    # predicts the probability of risk for each method.
    @staticmethod
    def analyze(path):
        from keras.models import load_model
        start = datetime.now()
        tok = Tokenizer(num_words=MAX_WORDS)
        vocab = pd.read_csv(os.path.realpath(__file__)[:-11] + '/data/java/vocab.csv')
        tok.fit_on_texts(vocab.input)
        root = os.path.realpath(__file__)[:-11] + "/data/java/checkpoints/"

        h5_ls, vuln_models = os.listdir(root), {}

        for h5 in h5_ls:
            progress = str(h5_ls.index(h5)+1) + "/" + str(len(h5_ls))
            print("\x1b[33m(" + progress + ") - Loading " + h5[:-3] + "...\x1b[m")
            vuln_models[h5[:-3]] = load_model(root + h5)

        jfile = JavaClass(path)
        for method in jfile.methods:
            print("\n\x1b[33mEvaluating " + method.name + "()...\x1b[m")
            metrics, i = [], 0
            for vuln_model in vuln_models:
                pred = float(vuln_models[vuln_model].predict(Javalect._embed(tok, str(method.tokens())))[0][0])
                metrics.append(pred)
            soft_metrics = list(softmax(np.asarray([metrics]))[0])
            print("  p-risk     p-dist     vulnerability")
            for vuln_model in vuln_models:
                print("  " + _fmt(metrics, i) + "     " + _fmt(soft_metrics, i) + "     " + vuln_model)
                i += 1

        print("\n\x1b[33mAnalyzed " + str(len(jfile.methods)) + " methods against " + str(len(h5_ls)) +
              " vulnerabilities in " + str(datetime.now() - start) + "\x1b[m.")
    
    @staticmethod
    def evaluate():
        """
        评估 data/java/dataset 中的每个数据集文件，使用对应的模型进行预测和评估。
        输出准确率和 F1 得分。
        """
        from model import AchillesModel

        # 定义数据集和模型路径
        dataset_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "java", "dataset")
        checkpoint_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "java", "checkpoints")

        # 检查目录是否存在
        if not os.path.isdir(dataset_dir):
            print(f"\x1b[31mDataset directory not found: {dataset_dir}\x1b[m")
            return
        if not os.path.isdir(checkpoint_dir):
            print(f"\x1b[31mCheckpoint directory not found: {checkpoint_dir}\x1b[m")
            return

        # 遍历数据集文件
        for dataset_file in os.listdir(dataset_dir):
            if dataset_file.endswith(".csv"):
                dataset_path = os.path.join(dataset_dir, dataset_file)
                model_name = dataset_file.replace(".csv", ".h5")
                model_path = os.path.join(checkpoint_dir, model_name)

                # 检查对应的模型文件是否存在
                if not os.path.isfile(model_path):
                    print(f"\x1b[31mModel not found for dataset {dataset_file}: {model_path}\x1b[m")
                    continue

                print(f"\x1b[33mEvaluating dataset: {dataset_file} with model: {model_name}\x1b[m")

                # 加载数据集
                try:
                    data = pd.read_csv(dataset_path)
                except Exception as e:
                    print(f"\x1b[31mError reading dataset {dataset_file}: {e}\x1b[m")
                    continue

                # 调用 AchillesModel.evaluate 进行评估
                try:
                    AchillesModel.evaluate(data, model_path)
                except Exception as e:
                    print(f"\x1b[31mError evaluating dataset {dataset_file} with model {model_name}: {e}\x1b[m")


    @staticmethod
    def continue_training(epochs):
        """
        使用data/java/dataset 中的每个数据集文件，继续训练对应的模型
        输出准确率和 F1 得分。
        """
        from model import AchillesModel

        # 定义数据集和模型路径
        dataset_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "java", "dataset")
        checkpoint_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "java", "checkpoints")

        # 检查目录是否存在
        if not os.path.isdir(dataset_dir):
            print(f"\x1b[31mDataset directory not found: {dataset_dir}\x1b[m")
            return
        if not os.path.isdir(checkpoint_dir):
            print(f"\x1b[31mCheckpoint directory not found: {checkpoint_dir}\x1b[m")
            return

        # 遍历数据集文件
        for dataset_file in os.listdir(dataset_dir):
            if dataset_file.endswith(".csv"):
                dataset_path = os.path.join(dataset_dir, dataset_file)
                model_name = dataset_file.replace(".csv", ".h5")
                model_path = os.path.join(checkpoint_dir, model_name)

                # 检查对应的模型文件是否存在
                if not os.path.isfile(model_path):
                    print(f"\x1b[31mModel not found for dataset {dataset_file}: {model_path}\x1b[m")
                    continue

                print(f"\x1b[33mTraining dataset: {dataset_file} with model: {model_name}\x1b[m")

                # 加载数据集
                try:
                    data = pd.read_csv(dataset_path)
                except Exception as e:
                    print(f"\x1b[31mError reading dataset {dataset_file}: {e}\x1b[m")
                    continue

                # 调用 AchillesModel.evaluate 进行评估
                try:
                    AchillesModel.continue_training(data, model_path,epochs)
                except Exception as e:
                    print(e)
                    print(f"\x1b[31mError training on {dataset_file} with model {model_name}: {e}\x1b[m")


# _fmt: truncates a probability of risk string `x` to 6 characters
# long. If the value of `x` is considerably small, we let "x-> -∞".
# We pass a list of risk probabilities `ls` to perform some last
# minute computation to determine, in this case, the maximum value,
# and inject an ANSI escape code to give it some color.
def _fmt(ls, x):
    if float(ls[x]) < 0.0001:
        return "x-> -∞"
    if ls[x] == max(ls):
        return "\x1b[33m" + str(ls[x])[0:6] + "\x1b[m"
    else:
        return str(ls[x])[0:6]
