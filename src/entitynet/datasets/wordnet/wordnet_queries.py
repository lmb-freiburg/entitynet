""" """

from packg.iotools import dump_json, load_json

from entitynet.paths import get_entitynet_annotations_dir

DEFAULT_KEY = "livingthings"
DEFAULT_KEY_PARENTS = "livingthingsandparents"


def load_wordnet_noun_queries(key=DEFAULT_KEY) -> list[str]:
    queries = load_json(get_entitynet_annotations_dir() / "wordnet/{key}.json")
    queries = list(map(str.lower, queries))
    return queries


def load_wordnet_noun_query2synnames_list(key=DEFAULT_KEY) -> dict[str, list[str]]:
    query2synnames_list = load_json(
        get_entitynet_annotations_dir() / f"wordnet/{key}-query2synnamelist.json"
    )
    query2synnames_list = {k.lower(): v for k, v in query2synnames_list.items()}
    return query2synnames_list


def load_wordnet_noun_query2synnames(key=DEFAULT_KEY) -> dict[str, str]:
    query2synnames_list = load_wordnet_noun_query2synnames_list(key=key)
    # break homonymies by taking the first source, should be an ok heuristic since through
    # alphabetic ordering the "n01" (main) synsets are first
    return {k: v[0] for k, v in query2synnames_list.items()}


def load_hierarchy(key=DEFAULT_KEY) -> dict[str, str]:
    """
    Simple hierarchy of synname -> parent_synname | None
    """
    hierarchy = load_json(get_entitynet_annotations_dir() / f"wordnet/hierarchy_{key}.json")
    return hierarchy


def load_full_hierarchy(key=DEFAULT_KEY) -> dict[str, list[str]]:
    """
    Simple hierarchy of synname -> [parent_synname, parent_parent_synname, ...]
    """
    hierarchy = load_json(get_entitynet_annotations_dir() / f"wordnet/hierarchy-full_{key}.json")
    return hierarchy


def invert_hierarchy(new_hierarchy):
    """
    Returns:
        Hierarchy of synname -> list of children_synname
    """
    inverted_hierarchy = dict()
    for synname, parent_synname in new_hierarchy.items():
        if parent_synname is None:
            continue
        if parent_synname not in inverted_hierarchy:
            inverted_hierarchy[parent_synname] = []
        if synname not in inverted_hierarchy:
            inverted_hierarchy[synname] = []
        inverted_hierarchy[parent_synname].append(synname)
    return inverted_hierarchy


def expand_hierarchy(new_hierarchy):
    def get_all_parents(key_):
        direct_parent = new_hierarchy[key_]
        if direct_parent is None:
            return []
        else:
            return [direct_parent] + get_all_parents(direct_parent)

    parents = {}
    for key, parent in new_hierarchy.items():
        if parent is None:
            parent_list = []
        else:
            parent_list = [parent] + get_all_parents(parent)
            parents[key] = parent_list

    dump_json(parents, "parents.json", indent=2)


def main():
    expand_hierarchy(load_hierarchy())


if __name__ == "__main__":
    main()
