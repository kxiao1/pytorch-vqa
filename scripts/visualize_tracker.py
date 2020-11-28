import sys
import torch
import matplotlib.pyplot as plt

def main():
    assert len(sys.argv) > 1
    log = torch.load(sys.argv[1], map_location=torch.device('cpu'))
    tracker = log['tracker']
    num_epochs = len(tracker['train_loss'])
    assert num_epochs == len(tracker['train_acc'])
    assert num_epochs == len(tracker['val_loss'])
    assert num_epochs == len(tracker['val_acc'])
    epoch_list = list(range(1, num_epochs+1))

    def get_means(L):
        return [sum(sublist) / len(sublist) for sublist in L]

    print(get_means(tracker['val_acc']))

    plt.plot(epoch_list, get_means(tracker['train_loss']), label='train loss')
    plt.plot(epoch_list, get_means(tracker['train_acc']), label='train acc')
    plt.plot(epoch_list, get_means(tracker['val_loss']), label='val loss')
    plt.plot(epoch_list, get_means(tracker['val_acc']), label='val acc')
    plt.legend(loc='best')
    plt.savefig('temp.jpg')

if __name__ == '__main__':
    main()
