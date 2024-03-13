import gzip
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("labels", type=str)
parser.add_argument("glosses", type=str)
parser.add_argument("--label_type", type=str, default="babel")
parser.add_argument("--negative_class", action="store_true", default=False)

args = parser.parse_args()

# if len(sys.argv) < 3:
#    print("ERROR. Usage: python3 align_glosses_with_labels.py labels.txt glosses.txt [babel/wndomains]")

# if len(sys.argv) == 4:
#    assert sys.argv[3] in ['babel', 'wndomains']
#    label_type = 1 if sys.argv[3] == 'babel' else 2
# else:
#    label_type = 1

label_type = {"babel": 1, "wndomains": 2}.get(args.label_type, 1)


with open(args.labels, "rt") as f:
    labels = {}
    for line in f:
        row = line.strip().split("\t")
        idx, label = row[0], row[label_type]
        if label == "None" and not args.negative_class:
            continue
        labels[idx] = label if label != "None" else "factotum"

with gzip.open(args.glosses, "rt", encoding="utf-8") as f:
    for line in f:
        row = line.split()
        idx, gloss = row[0], " ".join(row[1:])
        gloss = gloss.replace("\t", " ")
        try:
            print(f"{idx}\t{labels[idx]}\t{gloss}")
        except:
            continue
