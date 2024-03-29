"""
sparse retreival with bm25 using pyserini.
Input: query rewrite + query expansion
Output: trec file
Evaluation: pytrec_eval
! print_res: the way they deal with map and mrr scores from ndcg scores is referetable!
"""
from re import T
# from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')
import argparse
import os
from utils import check_dir_exist_or_build
from os import path
from os.path import join as oj
import toml
import numpy as np
import json
from pyserini.search.lucene import LuceneSearcher
import pytrec_eval


def main():
    args = get_args()
    
    query_list = []
    qid_list = []
    # read rewrites
    with open(args.input_query_path, "r") as f:
        data = f.readlines()

    # read expansions
    with open(args.input_query_path_2, "r") as f2:
        data_2 = f2.readlines()

    n = len(data)
    
    for i in range(n):
        data[i] = json.loads(data[i])
        if args.query_type == "raw":
            query = data[i]["query"]
        elif args.query_type == "rewrite":
            query = data[i]['output']
        elif args.query_type == "oracle":
            query = data[i]['oracle_query']
        elif args.query_type == "decode": #decode
            query = data[i]['oracle_utt_text']
            if args.eval_type == "answer":
                data_2[i] = json.loads(data_2[i])
                query = data_2[i]['answer_utt_text']
            elif args.eval_type == "oracle+answer":#oracle+answer
                data_2[i] = json.loads(data_2[i])
                query = query + ' ' + data_2[i]['answer_utt_text']#concatenation of rewrites and expansion
            
        query_list.append(query)
        qid_list.append(data[i]['sample_id'])
   
    # pyserini search
    searcher = LuceneSearcher(args.index_dir_path)
    searcher.set_bm25(args.bm25_k1, args.bm25_b)
    hits = searcher.batch_search(query_list, qid_list, k = args.top_k, threads = 40)

    # write search results to trec files
    with open(oj(args.output_dir_path, "bm25_QRIR_oracle+answer_res.trec"), "w") as f:
        for qid in qid_list:
            for i, item in enumerate(hits[qid]):
                f.write("{} {} {} {} {} {} {}".format(qid,
                                                "Q0",
                                                item.docid[3:],
                                                i+1,
                                                -i - 1 + 200, ## new score?
                                                item.score,
                                                "bm25"
                                                ))
                f.write('\n')


    res = print_res(oj(args.output_dir_path, "bm25_QRIR_oracle+answer_res.trec"), args.gold_qrel_file_path, args.rel_threshold)
    return res


def print_res(run_file, qrel_file, rel_threshold):
    """
    evaluate the run file using pytrec_eval
    """
    # read run file and gold qrel file
    with open(run_file, 'r' )as f:
        run_data = f.readlines()
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()
    
    qrels = {}
    qrels_ndcg = {}
    runs = {}
    
    for line in qrel_data:        
        # format: query, q0, passage, rel
        line = line.strip().split()
        query = line[0]
        passage = line[2]
        rel = int(line[3])
        if query not in qrels:
            qrels[query] = {}
        if query not in qrels_ndcg:
            qrels_ndcg[query] = {}
        # for NDCG
        qrels_ndcg[query][passage] = rel
        # for MAP, MRR, Recall
        if rel >= rel_threshold:
            rel = 1
        else:
            rel = 0
        qrels[query][passage] = rel
    
    for line in run_data:
        # format: qid, q0, docid, index, new score, score, 'bm25'
        line = line.split(" ")
        query = line[0]
        passage = line[2]
        rel = int(line[4])
        if query not in runs:
            runs[query] = {}
        runs[query][passage] = rel

    # pytrec_eval eval
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.5", "recall.10", "recall.20", "recall.100"})
    res = evaluator.evaluate(runs)
    map_list = [v['map'] for v in res.values()]
    mrr_list = [v['recip_rank'] for v in res.values()]
    recall_100_list = [v['recall_100'] for v in res.values()]
    recall_20_list = [v['recall_20'] for v in res.values()]
    recall_10_list = [v['recall_10'] for v in res.values()]
    recall_5_list = [v['recall_5'] for v in res.values()]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3"})
    res = evaluator.evaluate(runs)
    ndcg_3_list = [v['ndcg_cut_3'] for v in res.values()]

    res = {
            "MAP": np.average(map_list),
            "MRR": np.average(mrr_list),
            "NDCG@3": np.average(ndcg_3_list), 
            "Recall@5": np.average(recall_5_list),
            "Recall@10": np.average(recall_10_list),
            "Recall@20": np.average(recall_20_list),
            "Recall@100": np.average(recall_100_list),
        }

    
    logger.info("---------------------Evaluation results:---------------------")    
    logger.info(res)
    return res



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_query_path", type=str, default="output/qrecc/QR/test_QRIR_oracle_prefix.json")#"output/qrecc/QR/test_QRIR_oracle_prefix.json"
    parser.add_argument("--input_query_path_2", type=str, default="output/qrecc/QR/test_QRIR_answer_prefix.json")#"output/qrecc/QR/test_QRIR_answer_prefix.json"
    parser.add_argument("--index_dir_path", type=str, default="/home/wangym/data1/dataset/qrecc/indexes/bm25")#"datasets/qrecc/indexes/bm25"
    parser.add_argument("--output_dir_path", type=str, default="/home/wangym/data1/output/convgqr/qrecc/bm25")
    parser.add_argument("--gold_qrel_file_path", type=str, default="/home/wangym/data1/dataset/qrecc/new_preprocessed/qrecc_qrel.tsv")#"datasets/qrecc/new_preprocessed/qrecc_qrel.tsv"
    
    parser.add_argument("--query_type", type=str, default="decode")
    parser.add_argument("--eval_type", type=str, default="oracle+answer")
    parser.add_argument("--bm25_k1", type=float, default="0.82")
    parser.add_argument("--bm25_b", type=float, default="0.68")
    parser.add_argument("--top_k", type=int, default="100")
    parser.add_argument("--rel_threshold", type=int, default="1")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()
