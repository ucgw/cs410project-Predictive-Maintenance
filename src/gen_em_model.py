#!/usr/bin/env python3
"""
Usage:
    gen_em_model.py [--help] [--debug] [--show-viz] [--save-model] [--viz-words=<word_count>] [--duration=<duration>] [--topics=<topic>] [--iterations=<num_iterations>] <training_metads_file> <new_topic_tokens> <method>

Options:
  --topics=<topics>       number of topic clusters to generate
  --viz-words=<word_count>    number of top words to show around each topic
  --duration=<duration>   duration (in minutes) of new topic tokens event

Arguments:
  <training_metads_file>  filename with emtopic training metadata (in JSON)
  <new_topic_tokens>      string of topic tokens to suggest on
  <method>                string of topic modeling algorithm

Options:
  -h, --help    Show this screen and exit.
  --debug       Set debug flag for more output
"""
import sys
import os
import json
import math
import re
import numpy as np

from docopt import docopt
from operator import itemgetter
from typing import Dict, List, Tuple, Any

sys.path.insert(1, './lib')
import EMTopicModel as emtm
import VisualizeEMTopicModel as vemtm
import LdaLsaTopicModel as ldalsatm

DEFAULT_VIZ_WORD_COUNT = 5
DEFAULT_DURATION = 60
DEFAULT_NUM_ITERATIONS = 250

# https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture

# Add Tokens (strings) to ignore here for processing raw data
TOKEN_IGNORE = [
'and',
'for',
'not',
'are',
'can',
'till',
'non',
'over',
'from',
'the',
'when',
'that',
'only',
'all',
'out',
'ifications',
'some',
'quick',
'day',
'within',
'put',
'making',
'ker',
'cloudfl',
'aka',
'any',
'into',
'according',
'tor',
'mcp',
'rer',
'nel',
'need',
'tbd',
'remo',
'more',
'eir',
'rese',
'sent',
'inst',
'rine',
'encement',
'may',
'comes',
'exp',
'ify',
'bee',
'shd',
'ance',
'come',
'ocations',
'now',
'unt',
'breas',
'says',
'most',
'inea',
'well',
'rvation',
'with',
'acco',
'ing',
'unmo',
'str',
'add',
'ject1',
'umo',
'except',
'myxy',
'ths',
'manuy',
'moed',
'cess',
'til',
'but',
]

TOKEN_IGNORE_REGEX = re.compile('|'.join(TOKEN_IGNORE), re.I)


def emtopic_tokens_from_event(event_summary: str, min_len: int = 2) -> List[str]:
    """
    Extract tokens from an event summary

    Parameters:
    - event_summary (str): The summary of the event.
    - min_len (int): The minimum length of each token to consider. Defaults to 2.

    Returns:
    - List[str]: A list of tokens extracted from the event summary.
    """

    # remove TOKEN_IGNORE tokens from consideration
    event_summary = TOKEN_IGNORE_REGEX.sub('', event_summary)

    # next, strip out any characters that are non-alphanumeric
    # and replace with space; capturing more tokens
    event_summary = re.sub(r'[^a-zA-Z0-9\s\-]|\-', ' ', event_summary)

    # next, strip out minlen - 1 non-space tokens from
    # consideration and normalize to single spaces between
    # tokens
    minlen_rx = re.compile(r'\b[^\s+]{,'+f'{min_len-1}'+r'}\b')
    event_summary = re.sub('\s+', ' ', minlen_rx.sub('', event_summary))

    # next, strip out leading and trailing spaces from string
    # and lowercase everything
    event_summary = event_summary.lstrip()
    event_summary = event_summary.rstrip()
    event_summary = event_summary.lower()

    # finally, return the list of tokens from the summary
    print("Query Tokens Processed: {}".format(event_summary.split(' ')))
    return event_summary.split(' ')


def calc_max_logP_token(new_tokens: List[str], ordered_tokens: List[str], log_P_at_idx: np.ndarray,
                        weights: np.ndarray, debug: bool = False) -> Tuple[str, float]:
    """
    Calculate the token with the maximum weighted log probability.

    Parameters:
    - new_tokens (List[str]): List of new tokens.
    - ordered_tokens (List[str]): List of ordered tokens.
    - log_P_at_idx (np.ndarray): Array of log probabilities at the corresponding indices.
    - weights (np.ndarray): Array of weights for each token.
    - debug (bool): Whether to print debug information. Defaults to False.

    Returns:
    - Tuple[str, float]: A tuple containing the token with the maximum weighted log probability and
    the log probability value.
    """
    max_log_P_W_token = (None, -np.inf)

    for tok_idx, token in enumerate(new_tokens):
        try:
            idx = ordered_tokens.index(token)
        except ValueError:
            continue

        log_P_w = log_P_at_idx[idx]/weights[tok_idx]

        if debug is True:
            print(f"log_P weighted x{weights[tok_idx]}: {token},{log_P_w}")

        max_log_P_W_token = max(
            [max_log_P_W_token, (token, log_P_w)],
            key=itemgetter(1)
        )

    if debug is True:
        print(f"max log_P token: {max_log_P_W_token}")

    return max_log_P_W_token


def calc_max_token_hour_ops(token: str, dt_token_group_counts: Dict[str, Dict[str, int]]) -> \
        Tuple[int, int]:
    """
    Calculate the hour with the maximum frequency of a specific token.

    Parameters:
    - token (str): The token for which to calculate the maximum frequency hour.
    - dt_token_group_counts (Dict[str, Dict[str, int]]): Dictionary containing token counts grouped by hours.

    Returns:
    - Tuple[int, int]: A tuple containing the hour with the maximum frequency of
    the specified token and the frequency value.
    """
    max_hour_ops_freq = (np.inf, -np.inf)

    for dt_key, dt_val in dt_token_group_counts.items():
        if token not in dt_val.keys():
            continue

        tok_count = dt_token_group_counts[dt_key][token]
        max_hour_ops_freq = max(
            [max_hour_ops_freq, (dt_key, tok_count)],
            key=itemgetter(1)
        )

    return max_hour_ops_freq


def suggest_hour_ops_by_tokens(new_tokens: List[str], ordered_tokens: List[str],
                               dt_token_group_counts: Dict[str, Dict[str, int]], log_P_at_idx: np.ndarray,
                               debug: bool = False) -> List[Tuple[Tuple[str, float], Tuple[int, int]]]:
    """
    Suggest hour operations based on token probabilities.

    Parameters:
    - new_tokens (List[str]): List of new tokens.
    - ordered_tokens (List[str]): List of ordered tokens.
    - dt_token_group_counts (Dict[str, Dict[str, int]]): Dictionary containing token counts grouped by hours.
    - log_P_at_idx (np.ndarray): Array of log probabilities at the corresponding indices.
    - debug (bool): Flag to print debug information. Defaults to False.

    Returns:
    - List[Tuple[Tuple[str, float], Tuple[int, int]]]: A list containing tuples for suggestions.
      Each tuple includes the token with the maximum log probability and its corresponding hour operation.
    """
    num_toks = len(new_tokens)
    tok_OrderW = []
    tok_EvenW = [1.0]*num_toks

    for w in range(1, num_toks+1):
        tok_OrderW.append(num_toks/w)

    max_log_P_EvenW_token = calc_max_logP_token(new_tokens, ordered_tokens, log_P_at_idx, tok_EvenW, debug=debug)

    max_token = max_log_P_EvenW_token[0]

    evenW_token_hour_ops = calc_max_token_hour_ops(max_token, dt_token_group_counts)

    max_log_P_OrderW_token = calc_max_logP_token(new_tokens, ordered_tokens, log_P_at_idx, tok_OrderW, debug=debug)

    max_token = max_log_P_OrderW_token[0]

    orderW_token_hour_ops = calc_max_token_hour_ops(max_token, dt_token_group_counts)

    return [(max_log_P_EvenW_token, evenW_token_hour_ops),
            (max_log_P_OrderW_token, orderW_token_hour_ops)
           ]


def update_group_token_counts(curr: Dict[str, int], total: Dict[str, int]) -> Dict[str, int]:
    """
    Update the counts of tokens.

    Parameters:
    - curr (Dict[str, int]): The current counts of tokens.
    - total (Dict[str, int]): The total counts of tokens to update with.

    Returns:
    - Dict[str, int]: The updated counts of tokens.
    """
    new_total = {**curr, **total}
    for k, v in new_total.items():
        if k in curr and k in total:
            new_total[k] = v + curr[k]

    return new_total


def get_dt_slot_token_counts(metadata: List[List[Dict[str, List[str]]]]) -> Tuple[
    List[str], Dict[str, Dict[str, int]]]:
    """
    Get token counts grouped by date-time slots.

    Parameters:
    - metadata (List[List[Dict[str, Any]]]): Metadata containing information about request_id, hour_ops, tokens, etc.

    Returns:
    - Tuple[List[int], Dict[str, Dict[str, int]]]: A tuple containing the sorted list of time slots and
    a dictionary with token counts for each time slot.
    """

    dt_groups = set()
    dt_token_group_counts = {}

    for grouping in metadata:
        for record in grouping:
            rec_dt_group = set(record['hour_ops'])
            dt_groups = rec_dt_group.union(dt_groups)

            for hour in rec_dt_group:
                for tok in record['tokens']:
                    if hour in dt_token_group_counts:
                        try:
                            dt_token_group_counts[hour][tok] += 1
                        except KeyError:
                            dt_token_group_counts[hour][tok] = 1
                    else:
                        dt_token_group_counts[hour] = {}
                        dt_token_group_counts[hour][tok] = 1

    dt_groups = sorted(list(dt_groups))

    return dt_groups, dt_token_group_counts


def get_metadata(fname: str) -> List[List[Dict[str, Any]]]:
    """
    Read and parse metadata from a file.

    Parameters:
    - fname (str): The name of the file containing metadata.

    Returns:
    - List[List[Dict[str, Any]]]: Parsed metadata as a list of records.
    """
    metadata = None
    try:
        with open(fname) as tsfh:
            metadata = json.load(tsfh)
    except Exception as err:
        sys.stderr.write(f"Error parsing {fname}: {err}\n")
        sys.exit(1)
    return metadata


def get_token_mapping(counts: Dict[str, int]) -> Tuple[List[str], Dict[str, int]]:
    """
    Get the ordered tokens and a mapping of tokens to indices based on their counts.

    Parameters:
    - counts (Dict[str, int]): A dictionary containing token counts.

    Returns:
    - Tuple[List[str], Dict[str, int]]: A tuple containing the ordered tokens and a mapping of tokens to indices.
    """
    ordered_tokens = [k for k, v in sorted(counts.items(), reverse=True, key=lambda item: item[1])]
    tokmap = {tok: idx+1 for idx, tok in enumerate(ordered_tokens)}

    return ordered_tokens, tokmap


def transform_metadata_uci(metadata: List[List[Dict[str, Any]]]) -> \
        Tuple[List[str], Dict[str, Dict[str, int]], np.ndarray]:
    """
    Transform metadata into a format suitable for topic modeling analysis.

    Parameters:
    - metadata (List[List[Dict[str, Any]]]): Metadata.

    Returns:
    - Tuple[List[str], Dict[str, Dict[str, int]], np.ndarray]: A tuple containing ordered list of tokens,
    time slot token group counts, and a matrix X representing token counts in each time slot.
    """
    counts = {}
    group_counts = []

    dt_groups, dt_token_group_counts = get_dt_slot_token_counts(metadata)

    for group in dt_groups:
        curr_counts = dt_token_group_counts[group]
        counts = update_group_token_counts(curr_counts, counts)
        group_counts.append(curr_counts)

    ordered_tokens, tokmap = get_token_mapping(counts)

    X = np.zeros((len(dt_groups), len(ordered_tokens)))

    for idx, group in enumerate(group_counts):
        for token, count in group.items():
            X[idx-1, tokmap[token]-1] = count

    return ordered_tokens, dt_token_group_counts, X


if __name__ == '__main__':
    args = docopt(__doc__)
    tsdata = args['<training_metads_file>']
    model_type = args['<method>']
    cli_tokens = args['<new_topic_tokens>']
    debug = args['--debug'] or False

    showviz = args['--show-viz'] or False
    savemodel = args['--save-model'] or False

    try:
        N = int(args['--viz-words'])
    except Exception:
        N = DEFAULT_VIZ_WORD_COUNT

    try:
        duration = int(args['--duration'])
    except Exception:
        duration = DEFAULT_DURATION

    try:
        iterations = int(args['--iterations'])
    except Exception:
        iterations = DEFAULT_NUM_ITERATIONS

    new_tokens = emtopic_tokens_from_event(cli_tokens)

    metadata = get_metadata(tsdata)

    ordered_tokens, dt_token_group_counts, X = transform_metadata_uci(metadata)
    num_docs = X.shape[0]

    try:
        topics = int(args['--topics'])
    except Exception:
        topics = math.ceil(np.log(len(ordered_tokens) + len(new_tokens)))

    if model_type == "lsa":
        print("Topic Model Method: LSA")
        term_contributions, coherence_scores = ldalsatm.lsa(metadata, topics, new_tokens)
        for measure in coherence_scores:
            print(f"Coherence score ('{measure}' measure) for the LSA model with {topics} topics: "
                  f"{coherence_scores[measure]}")
        suggestions = suggest_hour_ops_by_tokens(new_tokens, ordered_tokens, dt_token_group_counts, term_contributions,
                                                 debug=debug)

        if debug is True:
            print(f"Documents: {num_docs}  Topic Clusters: {topics}")
            score_str = ", ".join([f"{coherence_scores[measure]} ({measure})" for measure in coherence_scores])
            print(f"Coherence scores for the LDA model with {topics} topics: {score_str}")

    elif model_type == "lda":
        print("Topic Model Method: LDA")
        pi, log_P, coherence_scores = ldalsatm.lda(metadata, topics, new_tokens)

        log_pi = np.log(pi)
        topic_idx, topic_prob = vemtm.get_top_topic_probability(log_pi)
        suggestions = suggest_hour_ops_by_tokens(new_tokens, ordered_tokens, dt_token_group_counts, log_P[topic_idx],
                                                 debug=debug)

        if debug is True:
            print(f"Documents: {num_docs}  Topic Clusters: {topics}")
            print(f"Probability Distributions:\n{log_pi}")
            print(f"Chosen Index: {topic_idx}  Probability: {topic_prob}")

            score_str = ", ".join([f"{coherence_scores[measure]} ({measure})" for measure in coherence_scores])
            print(f"Coherence scores for the LDA model with {topics} topics: {score_str}")

    else:
        print("Topic Model Method: EM")
        log_pi, log_P, log_W = emtm.run(X, topics, iterations=iterations, debug=debug)
        topic_idx, topic_prob = vemtm.get_top_topic_probability(log_pi)
        suggestions = suggest_hour_ops_by_tokens(new_tokens, ordered_tokens, dt_token_group_counts, log_P[topic_idx],
                                                 debug=debug)


        if debug is True:
            print(f"Documents: {num_docs}  Topic Clusters: {len(log_pi)}")
            print(f"Probability Distributions:\n{log_pi}")
            print(f"Chosen Index: {topic_idx}  Probability: {topic_prob}")
            print(f"Chosen Topic Cluster Weights:\n{log_W[:, topic_idx]}")


        if savemodel is not False:
            np.savez_compressed("em_topicmodel", X, log_pi, log_P, log_W)

    print(f"Suggestions:")
    for i, suggestion in enumerate(suggestions):
        print(f"Decision {i + 1}. Term: '{suggestion[0][0]}'. Hour: {suggestion[1][0]}. Frequency: {suggestion[1][1]} ")

    # LSA is based in reduction of dimensionality using SVD, it is not a probabilistic method, so
    # we can't visualize topic models with log probabilities
    if showviz is not False and model_type != "lsa":
        vemtm.viz_topic_freqs(ordered_tokens, log_pi, log_P, topics, topic_idx, N)
