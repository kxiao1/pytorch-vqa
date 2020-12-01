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

    mean_train_loss = get_means(tracker['train_loss'])
    mean_train_acc = get_means(tracker['train_acc'])
    mean_val_loss = get_means(tracker['val_loss'])
    mean_val_acc = get_means(tracker['val_acc'])

    plt.plot(epoch_list, mean_train_loss, label='train loss')
    plt.plot(epoch_list, mean_train_acc, label='train acc')
    plt.plot(epoch_list, mean_val_loss, label='val loss')
    plt.plot(epoch_list, mean_val_acc, label='val acc')
    plt.legend(loc='best')
    plt.savefig('temp.jpg')

    print("========SUMMARY========")
    best_index = min(range(num_epochs), key=lambda i: mean_val_loss[i])
    print("best val loss:", mean_val_loss[best_index])
    print("val acc @ best val loss:", mean_val_acc[best_index])
    smoothed = mean_val_acc[max(0,best_index-2):min(best_index+3, len(mean_val_acc))]
    print("smoothed val acc (average of 5) @ best val loss:", sum(smoothed)/len(smoothed))
    print("best val acc:", max(mean_val_acc))


if __name__ == '__main__':
    main()
