import os
import sys
import torch
import matplotlib
import matplotlib.pyplot as plt


def main():
    assert len(sys.argv) > 1
    logs = []
    for i in range(1, len(sys.argv)):
        logs.append(torch.load(sys.argv[i], map_location=torch.device('cpu')))

    log_name = os.path.basename(sys.argv[1])
    log_name = log_name[:log_name.index('-')]

    trackers = [log['tracker'] for log in logs]
    num_epochs = len(trackers[0]['train_loss'])

    def get_means(L):
        return [sum(sublist) / len(sublist) for sublist in L]

    tracker = {}
    print("Number of trackers:", len(trackers))
    for query_type in ["train_loss", "train_acc", "val_loss", "val_acc"]:
        # e.g. [[1, 2], [1.5, 2.5], [2, 3]]
        query_by_tracker = [get_means(tracker[query_type]) for tracker in trackers]
        print(query_type, query_by_tracker)

        res = []
        for i in range(len(query_by_tracker[0])):
            query_values = [query_by_tracker[j][i] for j in range(len(query_by_tracker))]
            mean = sum(query_values) / len(query_values)
            res.append(mean)
        tracker[query_type] = res
    
    # print(tracker)

    # assert num_epochs == len(tracker['train_acc'])
    # assert num_epochs == len(tracker['val_loss'])
    # assert num_epochs == len(tracker['val_acc'])
    epoch_list = list(range(1, num_epochs+1))

    mean_train_loss = tracker['train_loss']
    mean_train_acc = tracker['train_acc']
    mean_val_loss = tracker['val_loss']
    mean_val_acc = tracker['val_acc']

    plt.figure(figsize=(10,5))
    plt.subplots_adjust(wspace=0.3)

    plt.subplot(1, 2, 1)
    plt.plot(epoch_list, mean_train_loss, label='train')
    plt.plot(epoch_list, mean_val_loss, label='val')
    plt.legend(loc='best')
    plt.title(log_name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(epoch_list, mean_train_acc, label='train')
    plt.plot(epoch_list, mean_val_acc, label='val')
    plt.legend(loc='best')
    plt.title(log_name)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    print(f"save to {log_name}")
    plt.savefig(f"img/{log_name}.jpg")
    plt.show()

    print("========SUMMARY========")
    best_index = min(range(num_epochs), key=lambda i: mean_val_loss[i])
    print("best val loss:", mean_val_loss[best_index])
    print("val acc @ best val loss:", mean_val_acc[best_index])
    smoothed = mean_val_acc[max(0,best_index-2):min(best_index+3, len(mean_val_acc))]
    print("smoothed val acc (average of 5) @ best val loss:", sum(smoothed)/len(smoothed))
    last10 = mean_val_acc[-10:]
    last3 = mean_val_acc[-3:]
    print("average of last 10 val acc:", sum(last10)/len(last10))
    print("average of last 3 val acc:", sum(last3)/len(last3))
    print("best val acc:", max(mean_val_acc))


if __name__ == '__main__':
    main()
