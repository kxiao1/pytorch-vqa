import torch
import model
import data

def main():
    print("running on", "cuda:0" if torch.cuda.is_available() else "cpu")
    log = torch.load('logs/2017-08-04_00.55.19.pth', map_location=torch.device('cpu'))
    tokens = len(log['vocab']['question']) + 1

    net = torch.nn.DataParallel(model.Net(tokens))
    net.load_state_dict(log['weights'])

    dataset = data.get_loader(val=True)
    tracker = utils.Tracker()
    run(net, loader=dataset, optimizer=None, tracker=tracker, train=False, prefix='', epoch=0)


if __name__ == "__main__":
    main()
