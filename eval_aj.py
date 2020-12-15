import torch
import model
import utils
import data
import json
import colors
from model_baseline import run

def get_answer_map():
    res = {}
    res_q = {}
    with open("vizwiz/Annotations_all/val.json") as f:
        junk = json.loads(f.read())
        # print(junk)
        for dictionary in junk:
            image_name = dictionary["image"].split(".")[-2].split("_")[-1]
            answers = dictionary["answers"]
            res_q[int(image_name)] = dictionary["question"]
            res[int(image_name)] = [1 if answer["answer"] in colors.colors_set else 0 for answer in answers]
            res[int(image_name)] = 1 if sum(res[int(image_name)])/3 >= 1 else 0
    return res, res_q

def model_degenerate(question):
    if "color" in question or "Color" in question:
        return 1
    else:
        return 0

def main():
    print("running on", "cuda:0" if torch.cuda.is_available() else "cpu")
    true_answer_map, question_map = get_answer_map()
    log = torch.load('logs_is_color/final.pth', map_location=torch.device('cpu'))
    # log = torch.load('logs_is_color/2020-12-07_09:50:03.pth', map_location=torch.device('cpu'))
    # answer_map = {v: k for k, v in log['vocab']['answer'].items()}

    # print(log['eval'])
    print(log['eval']['answers'])
    
    temp = [1 if ans > 0.5 else 0 for ans in log['eval']['answers']]
    answ = temp
    accs = log['eval']['accuracies']
    temp = [torch.flatten(ans) for ans in log['eval']['idx']]
    idxs = [item for ans in temp for item in ans]
    print(idxs)

    # DEGENERATE model: just check if the word "color" is in the question
    degenerate_got_correct = 0
    got_correct = 0
    for ans, acc, idx in zip(answ, accs, idxs):
        if ans == true_answer_map[idx.item()]:
            got_correct += 1
        if model_degenerate(question_map[idx.item()]) == true_answer_map[idx.item()]:
            degenerate_got_correct += 1
        degen_answer = model_degenerate(question_map[idx.item()])
        print(f"index:{idx}, acc:{acc}, actual answer: {true_answer_map[idx.item()]}, predicted answer:{ans}, " +
            f"degenerate answer: {degen_answer}, question:{question_map[idx.item()]}")
    print("overall accuracy:", sum(accs)/len(accs))
    print(f"got correct: {got_correct}")
    print(f"degen got correct: {degenerate_got_correct}")
    print(f"total: {len(answ)}")


if __name__ == "__main__":
    main()
