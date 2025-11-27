"""
Build entitynet from URLS, step 3: count images, build tars.

See step 2 for an example of how the URL data looks like.

The following tasks are already done in the given URL dataframe:
Deduplication within the training set, text processing, deduplication with downstream tasks.
"""

import io
import os
from pathlib import Path

import pandas as pd
from attrs import define
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from webdataset.writer import TarWriter

from packg.iotools import dump_json, load_json
from packg.misc import uncollate
from packg.tqdmext import tqdm_max_ncols
from typedparser import TypedParser, add_argument
from visiontext.images import scale_image_smaller_side
from visiontext.iotools.single_tar_lookup import SingleTarLookup
from visiontext.mathutils import distribute_evenly

from entitynet.datasets.entitynet import N_SHARDS, EntityNetUrlBuildArgs, parse_args_for_url_build


@define
class EntityNetUrlBuildArgsTars(EntityNetUrlBuildArgs):
    write: bool = add_argument(
        action="store_true",
        help="Confirm writing tars",
    )
    delete_unfinished: bool = add_argument(
        shortcut="-d",
        action="store_true",
        help="Delete unfinished tars if found",
    )


MAX_SHORT_SIDE = 512  # scale images so that the smaller side is 512px or less


def main():
    parser = TypedParser.create_parser(EntityNetUrlBuildArgsTars, description=__doc__)
    args: EntityNetUrlBuildArgsTars = parser.parse_args()
    logger.info(f"{args}")
    entitynet_dir, split_list = parse_args_for_url_build(args)
    metadata_dir = entitynet_dir / "images_metadata"

    for split in split_list:
        logger.info(f"Processing split: {split}")
        parquets = sorted(list(metadata_dir.glob(f"{split}-*.parquet")))
        assert len(parquets) > 0, f"No parquets found for {split=} in {metadata_dir}"

        # collect downloaded images and metadata
        all_jpegs = []
        all_metadata = {}
        for parquet in parquets:
            logger.info(f"Process {split=} parquet {parquet.name}")
            df = pd.read_parquet(parquet)
            parquet_stem = parquet.stem
            download_dir = entitynet_dir / "images_raw" / parquet_stem
            jpegs = sorted(list(download_dir.glob("*.jpg")))
            metadata_dict = df.set_index("key").to_dict(orient="index")
            for jpeg in jpegs:
                key = jpeg.stem
                if key.endswith(".alpha"):
                    # some jpegs have .alpha.jpg suffix, but we only care about RGB
                    continue
                all_metadata[key] = metadata_dict[key]
                all_metadata[key]["img"] = jpeg.relative_to(entitynet_dir).as_posix()
                all_jpegs.append(jpeg)
        logger.info(f"Found total {len(all_jpegs)} jpegs for {split=}")
        assert len(all_jpegs) > 0, f"No jpegs found for {split}"
        if not args.write:
            logger.info(f"Skipping writing tars, pass --write to confirm writing")
            continue
            # files are in format {key}.jpg
        all_keys = list(all_metadata.keys())

        # compute the "temporary *.tar.json" files with metadata (what to write to tars)
        nimg_per_shard_file = entitynet_dir / f"num_images_per_shard-wds_{split}.json"
        metadata_file_format = entitynet_dir / f"temp_data_{split}" / "temporary_%05d.tar.json"
        n_shards = N_SHARDS[split]
        if not nimg_per_shard_file.is_file():
            logger.info(f"Compute metadata of shards for {split=}")
            n_datapoints = len(all_jpegs)
            per_shard = distribute_evenly(n_datapoints, n_shards)
            current_pos = 0
            for shard_id in tqdm_max_ncols(list(range(n_shards))):
                metadata_file = metadata_file_format.as_posix() % shard_id
                # get all images to be written to this shard
                end_pos = current_pos + per_shard[shard_id]
                shard_image_keys = all_keys[current_pos:end_pos]
                assert len(shard_image_keys) == per_shard[shard_id], (
                    f"{len(shard_image_keys)=} != {per_shard[shard_id]=} "
                    f"The tar would have a different number of images than expected. This is a bug."
                )
                # collect all the metadata for these images
                shard_metadata = {}
                for k in shard_image_keys:
                    v = all_metadata[k]
                    shard_metadata[k] = v
                dump_json(
                    shard_metadata, metadata_file, create_parent=True, verbose=False, indent=2
                )
                current_pos = end_pos
            dump_json(per_shard, nimg_per_shard_file, create_parent=True, verbose=False, indent=2)
            logger.info(f"Completed split: {split}")
        else:
            logger.info(f"Skipping writing {nimg_per_shard_file}, already exists")
            per_shard = load_json(nimg_per_shard_file)
            n_shards_file = len(per_shard)
            n_datapoints = sum(per_shard)
            sol = f"delete file {nimg_per_shard_file} and rerun."
            assert n_datapoints == len(all_jpegs), f"{n_datapoints}!={len(all_jpegs)}, {sol}"
            assert n_shards_file == n_shards, f"{n_shards_file}!={n_shards}, {sol}"

        if (entitynet_dir / f"data_{split}").is_dir():
            logger.error(
                f"Folder {entitynet_dir / f'data_{split}'} already exists. Not creating any tars. "
                f"This current tarwriting process is not implemented to be resumable. "
                f"Please restart."
            )
            continue
        # write the tars
        to_process = []
        for shard_id in list(range(n_shards)):
            metadata_file = metadata_file_format.as_posix() % shard_id
            tgt_tarfile = (entitynet_dir / f"data_{split}/shard_{shard_id:05d}.tar").as_posix()
            to_process.append((metadata_file, tgt_tarfile))
        logger.info(f"Prepared {len(to_process)} tar files to write for {split=}")
        logger.info(f"Files per shard: {per_shard}")

        ds = EntityNetUrlBuildTarMultiproc(
            entitynet_dir, to_process, delete_unfinished=args.delete_unfinished
        )
        pbar = tqdm_max_ncols(desc=f"Writing tars for {split}", total=len(ds))
        dl = DataLoader(ds, batch_size=1, num_workers=args.workers, collate_fn=uncollate)
        for i, _ in enumerate(dl):
            pbar.update()
        pbar.close()
        logger.info(f"Completed writing tars for {split=}")


class EntityNetUrlBuildTarMultiproc(Dataset):
    def __init__(self, entitynet_dir: Path, to_process: list, delete_unfinished: bool = False):
        self.entitynet_dir = entitynet_dir
        self.to_process = to_process
        self.delete_unfinished = delete_unfinished

    def __len__(self):
        return len(self.to_process)

    def __getitem__(self, index):
        metadata_file, tgt_tarfile = self.to_process[index]
        create_tar_shard(
            self.entitynet_dir,
            index,
            metadata_file,
            tgt_tarfile,
            delete_unfinished=self.delete_unfinished,
        )


def create_tar_shard(entitynet_dir, shard_id, metadata_file, tgt_tarfile, delete_unfinished=False):
    metadata = load_json(metadata_file)
    tgt_tarfile = Path(tgt_tarfile)
    tgt_jsonfile = tgt_tarfile.parent / f"metadata_{shard_id:05d}.json"

    if tgt_tarfile.is_file():
        verify_tar_shard(tgt_tarfile, tgt_jsonfile, metadata, delete_unfinished=delete_unfinished)
        return

    os.makedirs(tgt_tarfile.parent, exist_ok=True)
    output_tardata = {}
    logger.info(f"START writing {tgt_tarfile}")
    with TarWriter(tgt_tarfile.as_posix()) as fh:
        for imagenum, (imagekey, imagedata) in enumerate(metadata.items()):
            if imagenum % 1000 == 0:
                logger.info(f"{imagenum:05d}/{len(metadata):05d} in {tgt_tarfile.name}")
            imagefile = imagedata["img"]
            obj, metadata_update = process_single_image_for_tar(
                imagefile, imagekey, tgt_tarfile, entitynet_dir
            )
            fh.write(obj)
            output_tardata[imagekey] = {**imagedata, **metadata_update}
    # write new metadata, with image width and height updated
    dump_json(output_tardata, tgt_jsonfile, create_parent=True, verbose=False, indent=2)
    logger.info(f"DONE writing {tgt_tarfile}")


def verify_tar_shard(tgt_tarfile, tgt_jsonfile, metadata, delete_unfinished=False):
    logger.info(f"START verifying {tgt_tarfile}")
    # verify that the tar is correct
    error = False
    if tgt_jsonfile.is_file():
        # compare tar and metadata
        output_tardata = load_json(tgt_jsonfile)
        lookup = SingleTarLookup(tgt_tarfile.as_posix())
        files_in_tar = {f: None for f in lookup.get_filenames()}
        for input_imagekey in metadata.keys():
            imagekey_jpg = f"{input_imagekey}.jpg"
            if imagekey_jpg not in files_in_tar:
                logger.error(f"Missing {imagekey_jpg} in {tgt_tarfile}")
                error = True
            if input_imagekey not in output_tardata:
                logger.error(f"Missing {input_imagekey} in {tgt_jsonfile}")
                error = True
            if error:
                break
    else:
        # tar is not finished
        logger.error(f"Missing {tgt_jsonfile}")
        error = True
    if error:
        if delete_unfinished:
            logger.warning(f"Deleting unfinished shard {tgt_tarfile} and {tgt_jsonfile}")
            tgt_tarfile.unlink(missing_ok=True)
            tgt_jsonfile.unlink(missing_ok=True)
        else:
            raise RuntimeError(
                f"Errors found in {tgt_tarfile} use --delete_unfinished to automatically delete "
                f"all unfinished tars."
            )
    logger.info(f"DONE verifying {tgt_tarfile}")


def process_single_image_for_tar(imagefile, imagekey, tgt_tarfile, entitynet_dir):
    entitynet_dir = Path(entitynet_dir)
    img: Image.Image = Image.open(entitynet_dir / imagefile).convert("RGB")
    w, h = img.size
    if h > MAX_SHORT_SIDE or w > MAX_SHORT_SIDE:
        img = scale_image_smaller_side(img, MAX_SHORT_SIDE)
    img_no_metadata = Image.frombytes(img.mode, img.size, img.tobytes())
    neww, newh = img_no_metadata.size
    bio = io.BytesIO()
    img_no_metadata.save(bio, format="JPEG", quality=95)
    new_image_bytes = bio.getvalue()
    bio.close()
    img.close()
    img_no_metadata.close()

    obj = {
        "__key__": imagekey,
        "jpg": new_image_bytes,
    }
    metadata_update = {"height": newh, "width": neww}
    return obj, metadata_update


if __name__ == "__main__":
    main()
