from datasets import load_dataset

def create_dataset(args, tokenizer, split_dataset=True):

    print("[/] loading dataset...")
    dataset_name = args["dataset_name"]
    num_samples = args["num_dataset_samples"]
    dataset_shuffle_seed = args["dataset_shuffle_seed"]


    dataset = load_dataset(dataset_name, "1.0.0", split="all")
    # dataset = dataset.shuffle(seed=dataset_shuffle_seed).select(range(num_samples))
    # dataset = dataset.select(list(range(900, 925)))
    dataset = dataset.select(range(num_samples))
    print("[/] num_data_samples", len(dataset))

    def format_chat_template(row):
        # row_json = [{"role": "system", "content": row["instruction"]},
        # words = row["highlights"].split(" ")
        # num_words = 50#len(words)
        # print("[/] num_words", num_words)

        # row_json = [
                # {"role": "user", "content": f"Generate a summary(of not more than {num_words} words) for the following text:\n" + row["article"]},

                # {"role": "assistant", "content": "Summary:\n" + row["highlights"]}]

        row_json = [
            {"role": "user", "content": row["article"]},
            {"role": "assistant", "content": row["highlights"]}
        ]

        row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
        return row

    dataset = dataset.map(
        format_chat_template,
        num_proc= 4,
    )

    if split_dataset:
        test_size = args["test_size"]
        dataset = dataset.train_test_split(test_size=test_size, shuffle=False)

        for ds in dataset["train"]["text"][:10]:
            print(ds)

        # for ds in dataset["test"]["text"][:10]:
        #     print(ds)

    # else:
    #     for ds in dataset["text"][:10]:
    #         print(ds)

    return dataset
