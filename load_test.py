import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: enter degenerate/ baseline/ modified_attention/ big/ combined only")
        return
    name = sys.argv[1]
    if name == "degenerate":
        import model_degenerate
        Net = model_degenerate.Net
    elif name == "baseline":
        import model_baseline
        Net = model_baseline.Net
    elif name == "modified_attention":
        import model_modified_attention
        Net = model_modified_attention.Net
    elif name == "big":
        import model_big
        Net = model_big.Net
    elif name == "combined":
        import model_combined
        Net = model_combined.Net
    else:
        print("Invalid name, try again")
        return

    import data
    test_loader = data.get_loader(test=True)
    net = nn.DataParallel(Net(test_loader.dataset.num_tokens)).cuda()
    name = name+ datetime.now().strftime("-%Y-%m-%d_%H:%M:%S")
    target_name = os.path.join('logs_test', '{}.pth'.format(name))
    print('will save to {}'.format(target_name))
    # print("success")
    # assert(False)

if __name__ == '__main__':
    main()