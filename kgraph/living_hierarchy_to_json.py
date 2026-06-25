import json
import sys

if __name__ == "__main__":
    data = {}
    for line in sys.stdin:
        line = line.rstrip("\r\n")
        ent, parents = line.split("\t")
        parents = ["<" + p + ">" for p in parents[1:-1].split(";")]
        data[ent] = parents
    print(json.dumps(data, indent=2))
