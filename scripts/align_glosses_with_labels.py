import gzip
import sys

if len(sys.argv) < 3:
    print("ERROR. Usage: python3 align_glosses_with_labels.py labels.txt glosses.txt [babel/wndomains]")

if len(sys.argv) == 4:
    assert sys.argv[3] in ['babel', 'wndomains']
    label_type = 1 if sys.argv[3] == 'babel' else 2
else:
    label_type = 1


with open(sys.argv[1], 'rt') as f:
    labels = {}
    for line in f:
        row = line.strip().split('\t')
        idx, label = row[0], row[label_type]
        if label == 'None':
            continue
        labels[idx] = label

with gzip.open(sys.argv[2], 'rt', encoding='utf-8') as f:
    for line in f:
        row = line.split()
        idx, gloss = row[0], " ".join(row[1:])
        gloss = gloss.replace('\t', ' ')
        try:
            print(f"{idx}\t{labels[idx]}\t{gloss}")
        except:
            continue