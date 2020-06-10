import sys
import gzip

if len(sys.argv) < 3:
    print("ERROR. Usage: python3 align_glosses_with_labels.py labels.txt glosses.txt")

with open(sys.argv[1], 'rt') as f:
    labels = {}
    for line in f:
        row = line.split()
        idx, label = row[0], " ".join(row[1:])
        labels[idx] = label

with gzip.open(sys.argv[2], 'rt') as f:
    for line in f:
        row = line.split()
        idx, gloss = row[0], " ".join(row[1:])
        gloss = gloss.replace('\t', ' ')
        try:
            print(f"{idx}\t{labels[idx]}\t{gloss}")
        except:
            continue