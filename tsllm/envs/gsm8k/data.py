from datasets import load_dataset


def get_train_test_dataset(*args, **kwargs):
    num_train_data = kwargs.get("num_train_data", None)
    if num_train_data:
        train_dataset = load_dataset("gsm8k", "main", split=f"train[:{num_train_data}]")
    else:
        train_dataset = load_dataset("gsm8k", "main", split=f"train")
        # train_dataset = load_dataset("/mnt/afs/niuyazhe/code/LLM_Tree_Search/tsllm/envs/gsm8k/train_data")["train"] # TODO

    test_dataset = load_dataset("gsm8k", "main")["test"]

    # for debug
    # import pdb; pdb.set_trace()
    # 选择前两行作为测试集
    # test_dataset = train_dataset.select([0, 1])# TODO

    print(train_dataset, test_dataset)

    return train_dataset, test_dataset
