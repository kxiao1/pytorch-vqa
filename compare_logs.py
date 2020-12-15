import sys
import torch
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

# interpolate e.g. ([0, 1], 2) -> [0, 0.5, 1]
def interpolate(arr, n):
    res = []
    for i in range(len(arr) - 1):
        curr = arr[i]
        next_item = arr[i + 1]

        for j in range(0, n):
            res.append(curr * (1 - (j / n)) + next_item * (j / n))
    
    res.append(arr[-1])
    return res

# Format:
# [output file]
# [path 1] [scaling factor 1]
# [path 2] [scaling factor 2]
# ...
# [path n] [scaling factor n]
# where scaling factor = 4 if there are 4x as many augmented images
def main():
    output_path = sys.argv[1]

    # length 1 = [file, output_path, path1, scale1] -> 4
    # length 2 = [...] -> 6
    num_files = (len(sys.argv) // 2) - 1
    log_files = [torch.load(sys.argv[2 + 2*i]) for i in range(num_files)]
    scaling_factors = [int(sys.argv[3 + 2*i]) for i in range(num_files)]
    maximum_accuracies = [max(arr) for arr in log_files]
    # names = [sys.argv[2 + 2*i] for i in range(num_files)]
    names = ["Not augmented", "Augmented"][::-1]

    print(maximum_accuracies)

    val_accs = [torch.FloatTensor(log_files[i]['tracker']['val_acc']).mean(dim=1).numpy() for i in range(num_files)]

    print(val_accs)

    plt.figure()

    for i in range(num_files):
        xi = [i for i in range((len(val_accs[i]) - 1) * scaling_factors[i] + 1)]
        yi = interpolate(val_accs[i], scaling_factors[i])

        print(xi, yi)

        xi = xi[:30]
        yi = yi[:30]
        plt.plot(xi, yi, label = names[i])

    plt.plot()
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title("Validation accuracy of baseline model by data augmentation")
    plt.legend()
    plt.savefig('val_acc.png')

    # path = sys.argv[1]
    # results = torch.load(path)

    # val_acc = torch.FloatTensor(results['tracker']['val_acc'])
    # val_acc = val_acc.mean(dim=1).numpy()

    # plt.figure()
    # plt.plot(val_acc)
    # plt.savefig('val_acc.png')


if __name__ == '__main__':
    main()
