import cPickle as pkl
import os
from data_iterator import TextIterator

from main import (
    build_model,
    pred_probs,
    prepare_data,
    pred_acc,
    load_params,
    init_params,
    init_tparams,
)

# MUST MATCH the ids in `dic` in preprocess.py
id2label = ["entailment", "neutral", "contradiction"]


def main():
    model_name = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    model = "{}.npz".format(model_name)
    datasets = [
        "../../data/word_sequence/premise_snli_1.0_train.txt",
        "../../data/word_sequence/hypothesis_snli_1.0_train.txt",
        "../../data/word_sequence/label_snli_1.0_train.txt",
    ]

    valid_datasets = [
        "../../data/word_sequence/premise_snli_1.0_dev.txt",
        "../../data/word_sequence/hypothesis_snli_1.0_dev.txt",
        "../../data/word_sequence/label_snli_1.0_dev.txt",
    ]

    test_datasets = [
        "../../data/word_sequence/premise_snli_1.0_test.txt",
        "../../data/word_sequence/hypothesis_snli_1.0_test.txt",
        "../../data/word_sequence/label_snli_1.0_test.txt",
    ]
    dictionary = "../../data/word_sequence/vocab_cased.pkl"

    # load model model_options
    with open("%s.pkl" % model, "rb") as f:
        options = pkl.load(f)

    print(options)
    # load dictionary and invert
    with open(dictionary, "rb") as f:
        word_dict = pkl.load(f)

    n_words = options["n_words"]
    valid_batch_size = options["valid_batch_size"]

    valid = TextIterator(
        valid_datasets[0],
        valid_datasets[1],
        valid_datasets[2],
        dictionary,
        n_words=n_words,
        batch_size=valid_batch_size,
        shuffle=False,
    )
    test = TextIterator(
        test_datasets[0],
        test_datasets[1],
        test_datasets[2],
        dictionary,
        n_words=n_words,
        batch_size=valid_batch_size,
        shuffle=False,
    )

    # allocate model parameters
    params = init_params(options, word_dict)

    # load model parameters and set theano shared variables
    params = load_params(model, params)
    tparams = init_tparams(params)

    trng, use_noise, x1, x1_mask, x2, x2_mask, y, opt_ret, cost, f_pred = build_model(
        tparams, options
    )

    use_noise.set_value(0.0)
    valid_acc = pred_acc(f_pred, prepare_data, options, valid)
    test_acc = pred_acc(f_pred, prepare_data, options, test)

    print("valid accuracy", valid_acc)
    print("test accuracy", test_acc)

    predict_labels_valid = pred_label(f_pred, prepare_data, valid)
    predict_labels_test = pred_label(f_pred, prepare_data, test)

    with open("predict_gold_samples_valid.txt", "w") as fw:
        with open(valid_datasets[0], "r") as f1:
            with open(valid_datasets[1], "r") as f2:
                with open(valid_datasets[-1], "r") as f3:
                    for a, b, c, d in zip(predict_labels_valid, f3, f1, f2):
                        fw.write(
                            str(a)
                            + "\t"
                            + b.rstrip()
                            + "\t"
                            + c.rstrip()
                            + "\t"
                            + d.rstrip()
                            + "\n"
                        )

    with open("predict_gold_samples_test.txt", "w") as fw:
        with open(test_datasets[0], "r") as f1:
            with open(test_datasets[1], "r") as f2:
                with open(test_datasets[-1], "r") as f3:
                    for a, b, c, d in zip(predict_labels_test, f3, f1, f2):
                        fw.write(
                            str(a)
                            + "\t"
                            + b.rstrip()
                            + "\t"
                            + c.rstrip()
                            + "\t"
                            + d.rstrip()
                            + "\n"
                        )

    print("Done")


def pred_label(f_pred, prepare_data, iterator):
    labels = []
    for x1, x2, y in iterator:
        x1, x1_mask, x2, x2_mask, y = prepare_data(x1, x2, y)
        preds = f_pred(x1, x1_mask, x2, x2_mask)
        labels = labels + preds.tolist()

    return labels


if __name__ == "__main__":
    main()
