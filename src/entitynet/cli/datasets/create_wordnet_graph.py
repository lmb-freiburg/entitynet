"""
This script can be used to create wordnet graph visualization websites with d3.js

https://www.nltk.org/howto/wordnet.html

- Hypernyms = parents, hyponyms = children
- wnid has 8 numbers e.g. n00000001
- some of them do have more than 1 parent so its not a tree

{
    "type_parents": [],
    "inst_parents": [],
    "lemmas": ["entity"],
    "synname": "entity.n.01",
    "definition": "that which is perceived or known or inferred to have its own distinct existence (living or nonliving)",
    "wnid": "n00001740",
    "min_depth": 0,
    "max_depth": 0,
    "parent": None,
    "type_children": ["abstraction.n.06", "physical_entity.n.01", "thing.n.08"],
    "inst_children": [],
    "node_type": "root",
    "parents": [],
    "children": ["abstraction.n.06", "physical_entity.n.01", "thing.n.08"],
}

# to build smaller graphs than the full 70k nouns, pass the wordnet ids
# restrict to only living things:
python -m crx.run.wordnet.wordnet_graph -r crx/annotations/queries/livingthings-v2_synnames.json
# private/wordnet/livingthings-v1/index.html

# without restricting max leafs
python -m crx.run.wordnet.wordnet_graph -l 0


"""

import os
import shutil
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Optional

import networkx as nx
import numpy as np
from attrs import define
from loguru import logger
from nltk.corpus.reader import Synset  # noqa
from tqdm import tqdm

from packg.iotools import dumps_json, load_json
from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from typedparser import TypedParser, VerboseQuietArgs, add_argument

from entitynet.datasets.wordnet import load_wordnet_nouns


@define
class Args(VerboseQuietArgs):
    max_depth: int = add_argument(
        shortcut="-d", type=int, help="Max depth in one html file", default=6
    )
    max_leafs: int = add_argument(shortcut="-l", type=int, help="Max leafs to display", default=1)
    base_url: str = add_argument(
        shortcut="-b", type=str, help="Base url to use for the hrefs", default="."
    )
    restrict_synnames_file: Optional[Path] = add_argument(
        shortcut="-r",
        type=str,
        help="JSON file with list of wordnet synnames to restrict the graph to",
        default=None,
    )
    restrict_children_ids_file: Optional[Path] = add_argument(
        shortcut="-c",
        type=str,
        help="Restrict to these children ids and their parents",
        default=None,
    )
    output_dir: str = add_argument(
        shortcut="-o", type=str, help="Output dir for the html files", default="wordnet_graph"
    )


def get_href(wnid, base_url):
    return f"{base_url}/{wnid}.html"


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    logger.info(f"{args}")

    wordnet_data = load_wordnet_nouns()
    if args.restrict_synnames_file is not None:
        restrict_synnames = load_json(args.restrict_synnames_file)
        print(f"Restricting to {len(restrict_synnames)} leaves")
        # so now in v2 i only have leaves but need the parents for drawing the graph
        to_keep, to_check = set(), set()
        for synname in restrict_synnames:
            to_check.add(synname)

        while len(to_check) != len(to_keep):
            for synname in list(to_check):
                if synname in to_keep:
                    continue
                item = wordnet_data[synname]
                parents = item["type_parents"] + item["inst_parents"]
                for p in parents:
                    if p not in to_check:
                        to_check.add(p)
                to_keep.add(synname)
        wordnet_data = {k: wordnet_data[k] for k in sorted(to_keep)}
        print(f"Restricted to {len(wordnet_data)} ids")
    print(next(iter(wordnet_data.items())))

    # build full graph
    G = nx.DiGraph()
    root_id = None
    for i, (synname, data_dict) in enumerate(wordnet_data.items()):
        label = data_dict["lemmas"][0].replace("_", " ")
        is_leaf, is_root = True, False
        if len(data_dict["children"]) > 0:
            is_leaf = False
        parent_synnames = data_dict["type_parents"] + data_dict["inst_parents"]

        # after restriction some parents might be missing
        if args.restrict_synnames_file is not None:
            parent_synnames = [p for p in parent_synnames if p in wordnet_data]

        if len(parent_synnames) == 0:
            assert root_id is None, (
                f"Root already set to {root_id} " f"but found another root {synname} {label}"
            )
            root_id = synname
            is_root = True

        pruned_parents = ""
        if len(parent_synnames) == 0:
            parent_wnid = None
        elif len(parent_synnames) == 1:
            parent_wnid = parent_synnames[0]
        else:
            # in case of multiple parents, take the one with minimum depth, and remember the others
            parent_depths = [wordnet_data[p]["min_depth"] for p in parent_synnames]
            min_depth_index = np.argmin(parent_depths)
            parent_wnid = parent_synnames[min_depth_index]
            other_indices = [i for i in range(len(parent_synnames)) if i != min_depth_index]
            pruned_parent_wnids = [parent_synnames[i] for i in other_indices]
            pruned_parents = ",".join([wordnet_data[p]["lemmas"][0] for p in pruned_parent_wnids])

        G.add_node(
            synname,
            label=label,
            name=data_dict["lemmas"][0],
            is_leaf=is_leaf,
            is_root=is_root,
            pruned_parents=pruned_parents,
        )
        if parent_wnid is not None:
            G.add_edge(parent_wnid, synname)

    assert root_id is not None
    print(G)

    out_base_dir = Path(args.output_dir)
    target_xml = out_base_dir / f"tree_full.xml"
    if not target_xml.is_file():
        os.makedirs(target_xml.parent, exist_ok=True)
        nx.write_graphml(G, target_xml)
        print(f"Wrote graph to {target_xml}")

    # graph visualizing with D3
    this_dir = Path(__file__).absolute().parent
    source_html_file = this_dir / "d3.html"
    source_html_text = source_html_file.read_text()
    js_file = this_dir / "d3.min.js"

    max_depth = args.max_depth
    max_leafs = args.max_leafs
    subfolder = f"wordnet-d{max_depth}-l{max_leafs}"

    full_tree, level_counter = create_graph(
        G,
        wordnet_data,
        root_id,
        max_depth=max_depth,
        max_leafs=max_leafs,
        base_url=args.base_url,
    )
    out_dir = out_base_dir / f"{subfolder}"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(js_file, out_dir)
    pprint(level_counter)
    html_file = out_dir / f"index.html"

    legend_str = (
        "<b>Legend:</b><br/>green = leaf (no children),<br/>"
        "red = hidden leaves (not clickable),<br/>"
        "blue = parents with hidden children (clickable),<br/>"
        "white = parents with visible children (clickable)<br/>"
    )

    write_new_data(full_tree, level_counter, html_file, source_html_text, body=legend_str)
    print(f"Wrote {html_file.absolute()}")

    for synname, data_dict in tqdm(wordnet_data.items(), total=len(wordnet_data)):
        # if len(data_dict["children_wnids"]) == 0:
        #     continue
        tree, level_counter = create_graph(
            G,
            wordnet_data,
            synname,
            max_depth=max_depth,
            max_leafs=max_leafs,
            base_url=args.base_url,
        )
        html_file = out_dir / f"{synname}.html"
        title = f"{synname} ({data_dict['lemmas'][0]})"

        # get all parent wnids from the graph
        def _get_parents(c_wnid):
            c_parents = [c_wnid]
            all_parents = list(G.predecessors(c_wnid))
            if len(all_parents) == 0:
                return c_parents
            assert len(all_parents) == 1
            c_parent = all_parents[0]
            c_parents += _get_parents(c_parent)
            return c_parents

        parents = _get_parents(synname)
        head = """
<style>
table.thetable, table.thetable tr, table.thetable td, table.thetable th {
    border: 1px solid black; border-collapse: collapse;
}
table.thetable td {
    padding: 5px;
}
</style>
        """
        body_strs = [
            legend_str,
            "<table class='thetable'>",
            "<tr><th>D</th><th>synset</th><th>name</th><th>lemmas</th><th>definition</th></tr>",
        ]
        for i, parent in enumerate(parents):
            p_data = wordnet_data[parent]
            body_strs += [
                f"<tr>",
                f"<td><a href='{get_href(parent, args.base_url)}' target='_BLANK'>{parent}</a></td>",
                f"<td>{len(parents) - i}</td><td>{p_data['synname']}</td>",
                f"<td>{' | '.join(p_data['lemmas'])}</td>",
                f"<td>{p_data['definition']}</td></tr>",
            ]
        body_strs += ["</table>"]

        write_new_data(
            tree,
            level_counter,
            html_file,
            source_html_text,
            title=title,
            body="\n".join(body_strs),
            head=head,
        )


def create_graph(G, data, start_id="entity.n.01", max_depth=0, max_leafs=0, base_url="."):
    level_counter = defaultdict(int)
    start_level = 0

    # count total number of nodes in the graph when including all children
    def _count_children(current_id, current_level=0):
        ct = 1
        if G.nodes[current_id]["is_leaf"]:
            # real leaf
            return ct
        elif current_level >= max_depth > 0:
            # fake leaf because we stop propagating
            return ct
        else:
            # actual parent
            for child_id in G.successors(current_id):
                ct += _count_children(child_id, current_level + 1)
            return ct

    content_out = _count_children(start_id, 0)

    # disable leaf skipping for small graphs
    if content_out < 1000:
        max_leafs = 0

    def _add_children(current_id, current_level=0):
        level_counter[current_level] += 1
        item = data[current_id]
        children_content = []
        if G.nodes[current_id]["is_leaf"]:
            # real leaf
            color = f"33ff33"
        elif current_level >= max_depth > 0:
            # fake leaf because we stop propagating
            color = f"3333ff"
        else:
            # inbetween node where children are allowed to be displayed
            num_leafs = 0
            skipped_leafs = 0
            for child_id in G.successors(current_id):
                child_is_leaf = G.nodes[child_id]["is_leaf"] or current_level + 1 == max_depth
                if max_leafs > 0:
                    if child_is_leaf:
                        if num_leafs >= max_leafs:
                            skipped_leafs += 1
                            continue
                        else:
                            num_leafs += 1
                children_content.append(_add_children(child_id, current_level + 1))
            if skipped_leafs > 0:
                children_content.append(
                    {
                        "name": f"... ({skipped_leafs})",
                        "id": f"{current_id}_skipped",
                        "children": [],
                        "url": "none",
                        "color": "#770000",
                    }
                )
            color = f"ffffff"

        lemma = item["lemmas"][0]
        pruned_parents = G.nodes[current_id]["pruned_parents"]
        name_str = f"{lemma} ({pruned_parents})" if pruned_parents != "" else lemma
        content = {
            # "name": item["name"],
            "name": name_str,  # D{min_depth} P {parents}",
            "id": current_id,
            "children": children_content,
            "url": get_href(current_id, base_url),
        }
        content["color"] = f"#{color}"
        return content

    content_out = _add_children(start_id, start_level)
    return content_out, dict(sorted(level_counter.items(), key=lambda x: x[0]))


def write_new_data(tree, level_counter, target_file, source_html_text, title="", body="", head=""):
    """
    tree: data with format
        {
            "name": "entity",
            "id": "n1740",
            "parent": "null",
            "children": [
                {
                    "name": "abstraction",
                    "id": "n2137",
                    "parent": "entity"
                }
            ]
        }
    level_counter: number of nodes in a level of the graph
        {
            0: 1,
            1: 1
        }
    target_file:
    source_html_text:

    Returns:

    """
    for repa, repb in [
        ("TITLE_PLACEHOLDER_STRING", title),
        ("<!-- BODY_PLACEHOLDER -->", body),
        ("<!-- HEAD_PLACEHOLDER -->", head),
    ]:
        source_html_text = source_html_text.replace(repa, repb)

    # estimate required height of the tree
    height = max(level_counter.values()) / 230 * 7000
    width = len(level_counter) * 250
    data = [tree, width, height]
    json_string = dumps_json(data, indent=2)
    marker_begin = "<!-- BEGIN JSON DATA -->"
    marker_end = "<!-- END JSON DATA -->"
    first_half, temp = source_html_text.split(marker_begin)
    old_data, second_half = temp.split(marker_end)
    new_content = "\n".join(
        [
            first_half,
            marker_begin,
            '<div style="display:none" id="jsondata">',
            json_string,
            "</div>",
            marker_end,
            second_half,
        ]
    )
    target_file = Path(target_file)
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text(new_content)
    # logger.info(f"Wrote tree to {target_file}")


if __name__ == "__main__":
    main()
