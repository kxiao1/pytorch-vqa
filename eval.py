import torch
import model
import utils
import data
from train import run

def main():
    print("running on", "cuda:0" if torch.cuda.is_available() else "cpu")
    log = torch.load('logs/2017-08-04_00.55.19.pth', map_location=torch.device('cpu'))
    tokens = len(log['vocab']['question']) + 1
    answer_map = {v: k for k, v in log['vocab']['answer'].items()}

    net = torch.nn.DataParallel(model.Net(tokens))
    net.load_state_dict(log['weights'])

    dataset = data.get_loader(val=True)
    tracker = utils.Tracker()
    answ, accs, idxs = run(net, loader=dataset, optimizer=None, tracker=tracker, train=False, prefix='', epoch=0)
    for ans, acc, idx in zip(answ, accs, idxs):
        best_ans = answer_map[ans.item()]
        print(f"index:{idx} acc:{acc} answer:{best_ans}")
    print("overall accuracy:", sum(accs)/len(accs))


if __name__ == "__main__":
    main()
