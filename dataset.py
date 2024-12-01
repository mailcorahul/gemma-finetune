from datasets import load_dataset

def create_dataset(args, tokenizer):

    print("[/] loading dataset...")
    dataset_name = args["dataset_name"]
    num_samples = args["num_dataset_samples"]
    dataset_shuffle_seed = args["dataset_shuffle_seed"]
    test_size = args["test_size"]

    dataset = load_dataset(dataset_name, "1.0.0", split="all")
    dataset = dataset.shuffle(seed=dataset_shuffle_seed).select(range(num_samples)) # only use 1000 samples for demo

    def format_chat_template(row):
        # row_json = [{"role": "system", "content": row["instruction"]},
        row_json = [
                {"role": "user", "content": row["article"]},
                {"role": "assistant", "content": row["highlights"]}]
        row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
        return row

    dataset = dataset.map(
        format_chat_template,
        num_proc= 4,
    )

    dataset = dataset.train_test_split(test_size=test_size)
    return dataset