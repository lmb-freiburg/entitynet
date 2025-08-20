import datetime
from collections import defaultdict

from attrs import define
from loguru import logger
from natsort import natsorted

from packg.iotools import load_json
from packg.iotools.yamlext import load_yaml
from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from typedparser import TypedParser, VerboseQuietArgs, add_argument

from entitynet.paths import get_entitynet_output_dir


@define
class Args(VerboseQuietArgs):
    subdir: str = add_argument(shortcut="-s", default="all", help=f"Project subdirectory or 'all'")
    sort_by_time: bool = add_argument(shortcut="-t", action="store_true", help="Sort by time")


maxlen = 100
last_n = 5


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    logger.info(f"{args}")
    exp_dir = get_entitynet_output_dir() / "experiments"
    full_dir = exp_dir
    if args.subdir.lower() != "all":
        full_dir = full_dir / args.subdir
    print(f"{full_dir=}")
    assert full_dir.is_dir(), f"{full_dir=} is not a directory"

    lastjsons = natsorted(full_dir.glob("**/last.json"))
    sorter = defaultdict(list)
    maxlen = 0
    for lastjson in lastjsons:
        ts = lastjson.stat().st_mtime
        tf = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        content = load_json(lastjson)
        epoch = content["epoch"]
        runconfigyaml = lastjson.parent.parent / "runconfig.yaml"
        runconfig = load_yaml(runconfigyaml)
        max_epochs = runconfig["trainer"]["max_epochs"]
        if int(epoch) >= max_epochs - 1:
            sortstr = "done"
        else:
            sortstr = "running"
        relpath = lastjson.relative_to(exp_dir).parent.parent
        maxlen = max(maxlen, len(relpath.as_posix()))
        sorter[sortstr].append((int(epoch), relpath, ts, tf))

    for sort_key, sort_list in sorter.items():
        if args.sort_by_time:
            sort_list = list(sorted(sort_list, key=lambda x: x[2]))
        print(f"---------- {sort_key} ----------")
        for epoch, pth, ts, tf in sort_list:
            print(f"{epoch:02d} {pth.as_posix():{maxlen+1}s} {tf}")
        print()


if __name__ == "__main__":
    main()
