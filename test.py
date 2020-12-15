import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim

import config
import utils
import data


# python test.py type_of_model path_to_checkpoint
# e.g. python test.py baseline logs/baseline.pth
def main():
    assert len(sys.argv) == 3

    log = torch.load(sys.argv[2], map_location=torch.device('cpu'))
    num_tokens = len(log['vocab']['question']) + 1
    answer_map = {v: k for k, v in log['vocab']['answer'].items()}

    test_loader = data.get_loader(test=True)

    if sys.argv[1] == 'baseline':
        import model_baseline as model
    elif sys.argv[1] == 'modified_attention':
        import model_modified_attention as model
    elif sys.argv[1] == 'big':
        import model_big as model
    elif sys.argv[1] == 'combined':
        import model_combined as model
    elif sys.argv[1] == 'naive':
        import model_degenerate as model
        test_loader = data.get_loader(test=True, include_original_images=True)
    else:
        print("Re-enter the name of model!")
    net = nn.DataParallel(model.Net(num_tokens)).cuda()
    net.load_state_dict(log['weights'])

    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad], weight_decay=config.weight_decay)
    tracker = utils.Tracker()

    results = model.run(net, test_loader, optimizer, tracker, train=False, prefix='val', epoch=-1)

    anss, accs, idxs = results
    answers = [answer_map[ans.item()] for ans in anss]
    image_names = [f"VizWiz_test_{idx:08d}.jpg" for idx in idxs]
    results = [
        {"image": image_name, "answer": ans}
        for image_name, ans in zip(image_names, answers)
    ]

    assert sum(accs) == 0

    log_name = os.path.basename(sys.argv[2])
    log_name = log_name[:log_name.index('-')]

    with open(os.path.join("test_results", f"{log_name}.json"), 'w') as fd:
        json.dump(results, fd)


if __name__ == "__main__":
    main()
