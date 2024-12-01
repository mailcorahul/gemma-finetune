import torch
import numpy as np
import evaluate

def preprocess_logits_for_metrics(logits, labels):

    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels

def compute_metrics(eval_pred):

    # print(dir(eval_pred))

    # predictions, labels = eval_pred
    predictions = eval_pred.predictions.argmax(-1)
    labels = eval_pred.label_ids

    # print(predictions)
    # print(labels)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}
