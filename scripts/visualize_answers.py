import os, sys
import json
from collections import Counter
import itertools
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import data
import utils

def main():
    annotations_path = utils.path_for_annotations(val=True)

    with open(annotations_path, 'r') as fd:
        annotations_json = json.load(fd)

    answers = data.prepare_answers(annotations_json)

    all_tokens = itertools.chain.from_iterable(answers)
    counter = Counter(all_tokens)
    total_num_answers = sum(counter.values())
    print("total # of answers:", total_num_answers)
    most_common = counter.most_common()
    print("10 most common answers and counts", most_common[:10])
    counts = [pair[1] for pair in most_common]
    cdf = np.cumsum(counts) / total_num_answers
    plt.plot(list(range(1, len(most_common)+1)), cdf)
    plt.savefig('temp.jpg')
    for i, count in enumerate(counts):
        if count == 1:
            print(i)
            break
    print(f"counter[{i-1}]:", counts[i-1])
    print(f"counter[{i}]:", counts[i])
    print(cdf[1000])
    for pair in most_common[:400]:
        print(*pair)


if __name__ == '__main__':
    main()
