#
# Pyserini: Reproducible IR research with sparse and dense representations
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Most of the tokenization code here is copied from Facebook/DPR & DrQA codebase to avoid adding an extra dependency
"""

import os
import argparse
import copy
import json
import logging
import re
import unicodedata
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np

import regex

from qa_utils import SimpleTokenizer, has_answers

logger = logging.getLogger(__name__)


def evaluate_retrieval(
        retrieval_file: str, 
        topk: List[int], 
        regex: bool=False
    ) -> Dict[int, float]:
    """
    Evaluate QA Retrieval Recall@k.
    Input format: 
    [
        {
            "query": "...",
            "answers": ["...", "...", ... ],
            "contexts": [
                {
                    "id": "...", # passage id from database tsv file
                    "title": "",
                    "text": "....",
                    "score": "...",  # retriever score
                    "has_answer": true|false
        },
    ]
    """
    if retrieval_file.endswith(".json"):
        retrieval = json.load(open(retrieval_file))
    elif retrieval_file.endswith(".jsonl"):
        retrieval = list()
        with open(retrieval_file) as f:
            for line in f:
                retrieval.append(json.loads(line))
    else:
        raise NotImplementedError("Only receives `.json` or `.jsonl` inputs.")

    tokenizer = SimpleTokenizer()
    accuracy = { k : [] for k in topk }
    
    max_k = max(topk)
    hits = [0] * max_k


    for item in tqdm(retrieval):
        answers = item['answers']
        contexts = item['contexts']
        has_ans_idx = max_k  # first index in contexts that has answers
        all_hits = []
        for idx, ctx in enumerate(contexts):
            ctx['rank'] = idx + 1

            if idx >= max_k:
                break
            if 'has_answer' in ctx:
                if ctx['has_answer']:
                    all_hits.append(ctx['rank'])
            else:
                # text = ctx['text'].split('\n')[1]  # [0] is title, [1] is text
                text = ctx['text']
                if has_answers(text, answers, tokenizer, regex):
                    all_hits.append(ctx['rank'])
        
        item['all_hits'] = all_hits
        if len(all_hits) > 0:
            item['hit_min_rank'] = min(all_hits)
            has_ans_idx = min(all_hits) - 1
            hits[has_ans_idx:] = [v + 1 for v in hits[has_ans_idx:]]

        for k in topk:
            accuracy[k].append(0 if has_ans_idx >= k else 1)

    for k in accuracy.keys():
        accuracy[k] = np.mean(accuracy[k])

    for k in topk:
        print(f'Top{k}\taccuracy: {accuracy[k]:.4f}')

    for idx, h in enumerate(hits):
        hits[idx] = h / len(retrieval)
    
    return accuracy, hits, retrieval


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieval', type=str, metavar='path',
                        help="Path to retrieval output file.")
    parser.add_argument('--topk', type=int, nargs='+', default=[1, 3, 5, 10, 20, 50, 100], help="topk to evaluate")
    parser.add_argument('--regex', action='store_true', default=False, help="regex match")
    parser.add_argument('--save', type=str, default=None,
                        help="Path to output score files.")
    args = parser.parse_args()

    accuracy, hits, retrieval = evaluate_retrieval(args.retrieval, args.topk, args.regex)

    if args.save is not None:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        # with open(os.path.join(args.save, f"{args.prefix}result_topk.json"), 'w') as f:
        #     json.dump({"recall": accuracy}, f)

        with open(args.save.replace('.jsonl', '_hit@k.csv'), 'w') as f:
            for k, hit in enumerate(hits):
                f.write(f'{k+1},{hit}\n')
        print(f"Hit@k saved to {args.save.replace('.jsonl', '_hit@k.csv')}")

        with open(args.save, 'w') as f:
           for item in retrieval:
                f.write(json.dumps(item) + '\n')
        print(f"Retrieval results saved to {args.save}")
