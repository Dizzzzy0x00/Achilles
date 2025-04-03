import argparse
import os
from constants import *


def main():
    #print(version_info)

    # Top-level parser.
    parser = argparse.ArgumentParser(prog='achilles')
    subparsers = parser.add_subparsers(dest='subcommand', help='sub-command help')  # 添加 dest='subcommand'

    # achilles analyze file            =>   file='file'
    analyze = subparsers.add_parser('analyze', help='check a given file against vulnerability models')
    analyze.add_argument('file', type=str, help='analyze help')

    # achilles train java folder 30    =>    directory='folder', language='java', 'threshold=30'
    train = subparsers.add_parser('train', help='train models on class files in a given directory')
    train.add_argument('language', type=str, help='the language the corpus is written in')
    train.add_argument('directory', type=str, help='path to a folder containing corpus classes')
    train.add_argument('threshold', type=int, help='drop vulnerability classes with fewer example'
                                                   ' classes than the given threshold.')

    # achilles gen_data java folder 30 =>    directory='folder', language='java', 'threshold=30'
    gen_data = subparsers.add_parser('gen_data', help='generate datasets from class files in a given directory')
    gen_data.add_argument('language', type=str, help='the language the corpus is written in')
    gen_data.add_argument('directory', type=str, help='path to a folder containing corpus classes')
    gen_data.add_argument('threshold', type=int, help='drop vulnerability classes with fewer example'
                                                      ' classes than the given threshold.')

    # achilles ls java                 =>    language='java'
    ls = subparsers.add_parser('ls', help='view the list of trained models for a given language')
    ls.add_argument('language', type=str, help='the language to list vulnerability models for')

    continue_train = subparsers.add_parser('continue_train', help='continue training a model')
    continue_train.add_argument('language', type=str, help='the language the corpus is written in')
    continue_train.add_argument('epoch', type=int, help='the epoch to continue training from')

    # achilles evaluate test_data.csv  =>    test_data_path='test_data.csv'
    evaluate = subparsers.add_parser('evaluate', help='evaluate all trained models using datasets')
    evaluate.add_argument('language', type=str, help='the language the corpus is written in')
    
    args = parser.parse_args()

    if args.subcommand == 'analyze':  # analyze
        from javalect import Javalect
        if os.path.isfile(args.file):
            extension = "." + args.file.split(".")[-1]
            if extension in languages:
                if languages[extension] == "java":
                    Javalect.analyze(args.file)
                else:
                    print(f"\x1b[31mNo language support for \"{extension}\".\x1b[m")
            else:
                print(f"\x1b[31mNo language support for \"{extension}\".\x1b[m")
        else:
            print(f"\x1b[31mUnable to locate file: {args.file}\x1b[m")

    elif args.subcommand == 'train':  # train
        from javalect import Javalect
        if os.path.isdir(args.directory):
            if args.language == "java":
                print(f"\x1b[33mTraining {args.language} vulnerability models using files from \"{args.directory}\" "
                      f"with a threshold of {args.threshold}.\x1b[m")
                Javalect.train_models(args.directory, threshold=args.threshold)
            else:
                print(f"\x1b[31mLanguage \"{args.language}\" is not supported.\x1b[m")
        else:
            print(f"\x1b[31mUnable to locate folder: {args.directory}\x1b[m")

    elif args.subcommand == 'gen_data':  # gen_data
        from javalect import Javalect
        if os.path.isdir(args.directory):
            if args.language == "java":
                print(f"\x1b[33mGenerating datasets for {args.language} using files from \"{args.directory}\" "
                      f"with a threshold of {args.threshold}.\x1b[m")
                Javalect.gen_data(args.directory, threshold=args.threshold)
            else:
                print(f"\x1b[31mLanguage \"{args.language}\" is not supported.\x1b[m")
        else:
            print(f"\x1b[31mUnable to locate folder: {args.directory}\x1b[m")

    elif args.subcommand == 'ls':  # ls
        fpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", args.language, "checkpoints")
        if os.path.isdir(fpath):
            ls = os.listdir(fpath)
            print(f"\x1b[36mFound {len(ls)} {args.language} checkpoints:\x1b[m")
            for cwe in ls:
                print(f"  \x1b[36m*\x1b[m {cwe[:-3]}")
        else:
            print(f"\x1b[31mUnable to locate vulnerability models for \"{args.language}\".\x1b[m")
    

    elif args.subcommand == 'continue_train':  # continue_train
        from javalect import Javalect
        if args.language == "java":
            Javalect.continue_training(args.epoch)
        else:
            print(f"\x1b[31mLanguage \"{args.language}\" is not supported.\x1b[m")

    elif args.subcommand == 'evaluate':  # evaluate
        from javalect import Javalect
        if args.language == "java" :
            Javalect.evaluate()
        else:
            print(f"\x1b[31mLanguage \"{args.language}\" is not supported.\x1b[m")
        

if __name__ == "__main__":
    main()