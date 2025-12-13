import argparse
import json

try:
    import treelib
except ImportError:
    print("Please install treelib: pip install treelib")
    treelib = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="Data file (e.g. animals.hierarchy.tsv)")
    parser.add_argument("hierarchy", type=str, help="Hierarchy file (e.g. animals.hierarchy.json)")
    parser.add_argument("entity", type=str, help="Entity to find the hierarchy for (e.g. Q38280)")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    ent_to_label = {}
    with open(args.data) as f:
        for line in f:
            ent, label = line.strip().split("\t")[:2]
            ent_to_label[ent] = label[1:-4]

    with open(args.hierarchy) as f:
        ent_to_parents = json.load(f)
    if treelib is None:
        # trigger the error for real this time
        import treelib

    tree = treelib.Tree()

    ents = {None: [f"<http://www.wikidata.org/entity/{args.entity}>"]}
    depth = 0
    while len(ents) > 0:
        next_ents = {}
        occurences = {}
        for parent, ent in ents.items():
            for e in ent:
                occurences[e] = occurences.get(e, 0) + 1
                eid = f"{e}_{depth}_{occurences[e]}"
                tree.create_node(
                    tag=eid + " " + ent_to_label.get(e, e[1:-1].split("/")[-1]),
                    identifier=eid,
                    parent=parent,
                )
                if e in ent_to_parents:
                    next_ents[eid] = ent_to_parents[e]
        ents = next_ents
        depth += 1

    # print the tree in ascii
    print(tree.show(stdout=False, line_type="ascii"))


if __name__ == "__main__":
    main(parse_args())
