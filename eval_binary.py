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
            res[int(image_name)] = [1 if answer["answer"] == "unsuitable" or answer["answer"] == "unsuitable image" else 0 for answer in answers]
            res[int(image_name)] = 0 if sum(res[int(image_name)])/3 >= 1 else 1 
    return res

def main():
    print("running on", "cuda:0" if torch.cuda.is_available() else "cpu")
    true_answer_map = get_answer_map()
    log = torch.load('logs_karl/suitable_comp_2.pth', map_location=torch.device('cpu'))
    # answer_map = {v: k for k, v in log['vocab']['answer'].items()}
    
    temp = [torch.flatten(ans) for ans in log['eval']['answers']]
    answ = [item for ans in temp for item in ans]
    accs = log['eval']['accuracies']
    temp = [torch.flatten(ans) for ans in log['eval']['idx']]
    idxs = [item for ans in temp for item in ans]
    print(idxs)

    for ans, acc, idx in zip(answ, accs, idxs):
        print(f"index:{idx}, acc:{acc}, actual answer: {true_answer_map[idx.item()]}, predicted answer:{ans}")
    print("overall accuracy:", sum(accs)/len(accs))


if __name__ == "__main__":
    main()
