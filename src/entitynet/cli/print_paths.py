from entitynet.paths import (
    get_entitynet_annotations_dir,
    get_entitynet_output_dir,
    get_entitynet_repo_root,
)


def main():
    print(f"{get_entitynet_repo_root()=}")
    print(f"{get_entitynet_annotations_dir()=}")
    print(f"{get_entitynet_output_dir()=}")


if __name__ == "__main__":
    main()
