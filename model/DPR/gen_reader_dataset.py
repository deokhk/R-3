import ujson
import os
import logging
import argparse
import pathlib
from knockknock import slack_sender

from joblib import Parallel, delayed
from tqdm import tqdm
from dpr.options import setup_logger

logger = logging.getLogger()
setup_logger(logger)

def prepare_single_qapair(single_context):
    question = single_context["question"]
    answers = single_context["answers"]
    target = answers[0]
    ctxs = single_context["ctxs"]
    qa_pair ={
        "question": question,
        "target": target,
        "answers": answers,
        "ctxs":ctxs
    }
    return qa_pair


def prepare_datas(context_file, category, saved_path):
    logger.info(f"Now preparing reader {category} datasets")
    qa_pairs = []
    outputs = Parallel(n_jobs=-1, verbose=10)(delayed(prepare_single_qapair)(qapair) for qapair in tqdm(context_file))
    qa_pairs = [e for e in outputs]

    file = saved_path + f"/{category}.json"
    pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing results to {file}")
    with open(file, 'w') as f:
        ujson.dump(qa_pairs, f)

@slack_sender(webhook_url="https://hooks.slack.com/services/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", channel="knock_knock")
def main(args):
    paths = [args.train_context_path, args.dev_context_path, args.test_context_path]
    categories = ["train", "dev", "test"]
    for path, category in zip(paths, categories):
        logger.info(f"Start loading {category} context data")
        with open(path, 'r') as f:
            context = ujson.load(f)
        logger.info(f"Loaded {category} context data")
        prepare_datas(context, category, args.saved_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_context_path", help="path to the retrieval json train file")
    parser.add_argument("--dev_context_path", help="path to the retrieval dev json file")
    parser.add_argument("--test_context_path", help="path to output train file in csv format")
    parser.add_argument("--saved_path", help="path where the reader training dataset will be saved")
    args = parser.parse_args()

    main(args)