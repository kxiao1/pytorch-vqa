import torch
import model
import utils
import data
import json
from train import run

def get_answer_map():
    res = {}
    with open("vizwiz/Annotations_all/val.json") as f:
        junk = json.loads(f.read())
        # print(junk)
        for dictionary in junk:
            image_name = dictionary["image"].split(".")[-2].split("_")[-1]
            answers = dictionary["answers"]
            res[int(image_name)] = [answer["answer"] for answer in answers]
    return res

def main():
    print("running on", "cuda:0" if torch.cuda.is_available() else "cpu")
    true_answer_map = get_answer_map()
    # log = torch.load('logs/baseline.pth', map_location=torch.device('cpu'))
    log = torch.load('logs_karl/karl_modified_3000_no_decay.pth', map_location=torch.device('cpu'))
    tokens = len(log['vocab']['question']) + 1
    answer_map = {v: k for k, v in log['vocab']['answer'].items()}
    answ = log['eval']['answers']
    accs = log['eval']['accuracies']
    idxs = log['eval']['idx']
    colors = set([
        'red',
        'blue',
        'yellow',
        'green',
        'black',
        'white',
        'grey',
        'purple',
        'pink',
        'beige',
        'brown',
        'orange',
        'gold',
        'silver',
        'teal',
        'turquoise',
    ])
    color_accs = []
    for ans, acc, idx in zip(answ, accs, idxs):
        best_ans = answer_map[ans.item()]
        am = true_answer_map[idx.item()]
        if len(colors.intersection(am)) > 0:
            color_accs.append(acc)
            print(f"index:{idx}, acc:{acc}, actual answer: {true_answer_map[idx.item()]}, predicted answer:{best_ans}")
    print("overall accuracy:", sum(accs)/len(accs))
    print("overall color accuracy:", sum(color_accs)/len(color_accs))


if __name__ == "__main__":
    main()
