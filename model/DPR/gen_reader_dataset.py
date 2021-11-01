import ujson
import os
import logging
import pathlib
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


def prepare_datas(context_file, category):
    logger.info(f"Now preparing reader {category} datasets")
    qa_pairs = []
    outputs = Parallel(n_jobs=-1, verbose=10)(delayed(prepare_single_qapair)(qapair) for qapair in tqdm(context_file))
    qa_pairs = [e for e in outputs]

    file = os.getcwd() + f"/reader_dataset/{category}.json"
    pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing results to {file}")
    with open(file, 'w') as f:
        ujson.dump(qa_pairs, f)

def main():
    train_context_path = "/home/deokhk/research/MultiQA/model/DPR/retrieval_eval_output/context_for_reader/c_reader_train.json"
    dev_context_path = "/home/deokhk/research/MultiQA/model/DPR/retrieval_eval_output/context_for_reader/c_reader_dev.json"
    test_context_path = "/home/deokhk/research/MultiQA/model/DPR/retrieval_eval_output/from_new_hn_ckpt.json"

    paths = [train_context_path, dev_context_path, test_context_path]
    categories = ["train", "dev", "test"]
    for path, category in zip(paths, categories):
        logger.info(f"Start loading {category} context data")
        with open(path, 'r') as f:
            context = ujson.load(f)
        logger.info(f"Loaded {category} context data")
        prepare_datas(context, category)
    
if __name__ == '__main__':
    main()