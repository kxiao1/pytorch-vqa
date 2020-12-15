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
import colors

def main():
    annotations_path = utils.path_for_annotations(val=True)

    with open(annotations_path, 'r') as fd:
        annotations_json = json.load(fd)

    answers = data.prepare_answers(annotations_json)

    all_tokens = itertools.chain.from_iterable(answers)
    counter = Counter(all_tokens)
    total_num_answers = sum(counter.values())
    print("total # of answers:", total_num_answers)
    print("total # of distinct answers:", len(counter))
    most_common = counter.most_common()
    print("10 most common answers and counts", most_common[:10])
    counts = [pair[1] for pair in most_common]
    cdf = np.cumsum(counts) / total_num_answers
    plt.plot(list(range(1, len(most_common)+1)), cdf)
    plt.xlabel("Top N answers")
    plt.ylabel("Proportion of all answers")
    plt.title("Distribution of Answers")
    plt.savefig('temp.jpg')
    for i, count in enumerate(counts):
        if count == 1:
            print(i)
            break
    print(f"counter[{i-1}]:", counts[i-1], cdf[i-1])
    print(f"counter[{i}]:", counts[i], cdf[i])
    print(cdf[1000])
    for pair in most_common[:100]:
        print(pair[0], pair[1], pair[1]/total_num_answers)
    num_color_answers = 0
    for color in colors.colors:
        if color in counter:
            num_color_answers += counter[color]
    print("num color answers", num_color_answers, num_color_answers/total_num_answers)
        


if __name__ == '__main__':
    main()
