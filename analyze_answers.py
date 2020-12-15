import json

def main():
    with open("vizwiz/Annotations_all/train.json") as f:
        raw_data = f.read()
        training_examples = json.loads(raw_data)

    res = {}
    res_m = {}
    for entry in training_examples:
        answers = []

        for answer in entry["answers"]:
            answers.append(answer["answer"])
        distinct_answers = set(answers)

        max_agreement = 0
        for distinct_answer in distinct_answers:
            max_agreement = max(max_agreement, sum([1 if a == distinct_answer else 0 for a in answers]))

        num_answers = len(distinct_answers)
        if num_answers in res:
            res[num_answers] += 1
        else:
            res[num_answers] = 1

        if max_agreement in res_m:
            res_m[max_agreement] += 1
        else:
            res_m[max_agreement] = 1
        
    print(res)
    print(res_m)

if __name__ == "__main__":
    main()