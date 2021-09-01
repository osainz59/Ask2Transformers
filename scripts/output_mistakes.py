from argparse import ArgumentParser
import numpy as np


parser = ArgumentParser("output_mistakes")

parser.add_argument(
    "-p",
    "--predictions",
    type=str,
    default="experiments/splitted_topics/output.npy",
    help="System predictions.",
)
parser.add_argument(
    "-l",
    "--labels",
    type=str,
    default="experiments/splitted_topics/labels.npy",
    help="Labels.",
)
parser.add_argument("--label_names", type=str, default="data/babel_topics.txt", help="Domain labels.")
parser.add_argument(
    "--glosses",
    type=str,
    default="data/babeldomains.domain.gloss.tsv",
    help="Test file.",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="experiments/splitted_topics/wrong_predictions.txt",
    help="Output file containing wrong predictions.",
)


def main(opt):
    with open(opt.label_names, "rt") as f:
        topics = np.array([topic.rstrip().replace("_", " ") for topic in f])

    with open(opt.glosses, "rt") as f:
        info = np.array([line.rstrip().split("\t") for line in f])

    output = np.load(opt.predictions)
    labels = np.load(opt.labels)

    output_ = np.argmax(output, axis=-1)
    wrong_predictions = labels != output_

    assert output[wrong_predictions].shape[0] == labels[wrong_predictions].shape[0]

    with open(opt.output, "wt") as f:

        for (sense_id, l, gloss), out, true in zip(
            info[wrong_predictions],
            output[wrong_predictions],
            labels[wrong_predictions],
        ):
            f.write(f"Sense-id:\t{sense_id}\n")
            f.write(f"Gloss:\t\t{gloss}\n")
            f.write(f"Correct label:\t{l}\n")
            f.write(f"Top-5 Predictions:\n")
            idxs = out.argsort()[::-1]
            for i in range(5):
                f.write(f"   {out[idxs[i]]:.4f}\t{topics[idxs[i]]}\n")
            f.write("\n")


if __name__ == "__main__":
    opt = parser.parse_args()
    main(opt)
