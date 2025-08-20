import json
import os
import warnings
from pathlib import Path
from subprocess import call

import torch
from torch.utils.data import default_collate
from torchvision.datasets import (
    CIFAR10,
    CIFAR100,
    DTD,
    GTSRB,
    MNIST,
    PCAM,
    STL10,
    CocoCaptions,
    Country211,
    EuroSAT,
    FGVCAircraft,
    Flowers102,
    Food101,
    ImageFolder,
    ImageNet,
    OxfordIIITPet,
    RenderedSST2,
    StanfordCars,
)

from packg.log import logger

from clip_benchmark.datasets.imagenet_wnids import (
    ALL_IMAGENET_WORDNET_IDS,
    IMAGENET_A_WNIDS,
    IMAGENET_O_WNIDS,
    IMAGENET_R_WNIDS,
)
from entitynet.datasets.inat19 import iNat19
from entitynet.datasets.sun397_cached import SUN397CachedPaths

from . import (
    babel_imagenet,
    caltech101,
    flickr,
    imagenetv2,
    objectnet,
    sugar_crepe,
    voc2007,
    winoground,
)


def build_dataset(
    dataset_name,
    root="root",
    transform=None,
    split="test",
    download=True,
    annotation_file=None,
    language="en",
    task="zeroshot_classification",
    wds_cache_dir=None,
    custom_classname_file=None,
    custom_template_file=None,
    **kwargs,
):
    """
    Main function to use in order to build a dataset instance,

    dataset_name: str
        name of the dataset

    root: str
        root folder where the dataset is downloaded and stored. can be shared among datasets.

    transform: torchvision transform applied to images

    split: str
        split to use, depending on the dataset can have different options.
        In general, `train` and `test` are available.
        For specific splits, please look at the corresponding dataset.

    annotation_file: str or None
        only for datasets with captions (used for retrieval) such as COCO
        and Flickr.

    custom_classname_file: str or None
        Custom classname file where keys are dataset names and values are list of classnames.

    custom_template_file: str or None
        Custom template file where keys are dataset names and values are list of prompts, or dicts
        where keys are classnames and values are class-specific prompts.

    """
    use_classnames_and_templates = task in ("zeroshot_classification", "linear_probe")
    if use_classnames_and_templates:  # Only load templates and classnames if we have to
        current_folder = os.path.dirname(__file__)

        # Load <LANG>_classnames.json (packaged with CLIP benchmark that are used by default)
        default_classname_file = os.path.join(current_folder, language + "_classnames_nodups.json")
        # default_classname_file = os.path.join(current_folder, language + "_classnames.json")
        if os.path.exists(default_classname_file):
            with open(default_classname_file, "r") as f:
                default_classnames = json.load(f)
        else:
            default_classnames = None

        # Load <LANG>_zeroshot_classification_templates.json  (packaged with CLIP benchmark that are used by default)
        default_template_file = os.path.join(
            current_folder, language + "_zeroshot_classification_templates.json"
        )
        if os.path.exists(default_template_file):
            with open(default_template_file, "r") as f:
                default_templates = json.load(f)
        else:
            default_templates = None

        # Load custom classnames file if --custom_classname_file is specified
        if custom_classname_file:
            if not os.path.exists(custom_classname_file):
                custom_classname_file = os.path.join(current_folder, custom_classname_file)
            assert os.path.exists(
                custom_classname_file
            ), f"Custom classname file '{custom_classname_file}' does not exist"
            with open(custom_classname_file, "r") as f:
                custom_classnames = json.load(f)
        else:
            custom_classnames = None

        # Load custom template file if --custom_template_file is specified
        if custom_template_file:
            if not os.path.exists(custom_template_file):
                # look at current_folder
                custom_template_file = os.path.join(current_folder, custom_template_file)
            assert os.path.exists(
                custom_template_file
            ), f"Custom template file '{custom_template_file}' does not exist"
            with open(custom_template_file, "r") as f:
                custom_templates = json.load(f)
        else:
            custom_templates = None

    def download_imagenet(r):
        os.makedirs(r, exist_ok=True)
        call(
            f"wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz --output-document={r}/ILSVRC2012_devkit_t12.tar.gz",
            shell=True,
        )
        call(
            f"wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --output-document={r}/ILSVRC2012_img_val.tar",
            shell=True,
        )

    train = split == "train"

    if dataset_name.startswith("inat19"):
        if dataset_name.endswith("-latin"):
            language = "latin"
        else:
            language = "en"
        ds = iNat19(split=split, transform=transform, language=language)
    elif dataset_name == "cifar10":
        assert split in (
            "train",
            "test",
        ), f"Only `train` and `test` split available for {dataset_name}, got '{split}'"
        ds = CIFAR10(root=root, train=train, transform=transform, download=download, **kwargs)
    # # moved below to clip_benchmark_extension.py
    elif dataset_name == "cifar100":
        assert split in (
            "train",
            "test",
        ), f"Only `train` and `test` split available for {dataset_name}"
        ds = CIFAR100(root=root, train=train, transform=transform, download=download, **kwargs)
    elif dataset_name == "imagenet1k":
        if split == "val":
            print(
                f"WARN: Setting imagenet1k split to test from val for clip_benchmark compatibility."
            )
            split = "test"
        assert split in (
            "train",
            "test",
        ), f"Only `train` and `test` split available for {dataset_name}"
        if not os.path.exists(root):
            print(f"WARN: Downloading imagenet to {root}")
            download_imagenet(root)
        ds = ImageNet(root=root, split="train" if train else "val", transform=transform, **kwargs)
        ds.classes = default_classnames["imagenet1k"]
    elif dataset_name == "imagenet-w":
        assert split in (
            "train",
            "test",
        ), f"Only `train` and `test` split available for {dataset_name}"
        from imagenet_w import AddWatermark
        from torchvision.transforms import CenterCrop, Normalize

        if not os.path.exists(root):
            download_imagenet(root)
        index_normalize = None
        crop_size = None
        if transform is not None:
            for i, t in enumerate(transform.transforms):
                if isinstance(t, Normalize):
                    index_normalize = i
                elif isinstance(t, CenterCrop):
                    crop_size = min(t.size)
            assert crop_size is not None, "CenterCrop not found in transform"
            assert index_normalize is not None, "Normalize not found in transform"
            transform.transforms.insert(index_normalize, AddWatermark(crop_size))
        else:
            print(f"WARN: No transform given so imagenet-w cannot add the watermark transform.")
        ds = ImageNet(root=root, split="train" if train else "val", transform=transform, **kwargs)
        # ds.classes = custom_classnames["imagenet1k"]  # bug
        ds.classes = default_classnames["imagenet1k"]
    elif dataset_name == "babel_imagenet":
        assert split in (
            "train",
            "test",
        ), f"Only `train` and `test` split available for {dataset_name}"
        # babel ImageNet from https://github.com/gregor-ge/Babel-ImageNet
        if not os.path.exists(root):
            download_imagenet(root)
        classnames = json.load(open(os.path.join(current_folder, "babel_imagenet.json")))
        assert (
            language.upper() in classnames
        ), f"Language '{language}' not supported for Babel-ImageNet"
        classnames = classnames[language.upper()]
        templates = json.load(open(os.path.join(current_folder, "nllb_dist13b_prompts.json")))
        templates = templates[language.upper()]
        templates = [t.replace("{}", "{c}") for t in templates]
        idxs, classnames = classnames
        ds = babel_imagenet.BabelImageNet(
            root=root, idxs=idxs, split="train" if train else "val", transform=transform, **kwargs
        )
        ds.classes = classnames
        ds.templates = templates
    elif dataset_name == "imagenet1k-unverified":
        assert split in (
            "train",
            "test",
        ), f"Only `train` and `test` split available for {dataset_name}"
        split = "train" if train else "val"
        ds = ImageFolder(root=os.path.join(root, split), transform=transform, **kwargs)
        # use classnames from OpenAI
        ds.classes = default_classnames["imagenet1k"]
    elif dataset_name == "imagenetv2":
        assert split == "test", f"Only `test` split available for {dataset_name}"
        os.makedirs(root, exist_ok=True)
        ds = imagenetv2.ImageNetV2Dataset(
            variant="matched-frequency", transform=transform, location=root
        )
        ds.classes = default_classnames["imagenet1k"]
    elif dataset_name == "imagenet_sketch":
        assert split == "test", f"Only `test` split available for {dataset_name}"
        # Downloadable from https://drive.google.com/open?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA
        # or https://huggingface.co/datasets/songweig/imagenet_sketch/resolve/main/data/ImageNet-Sketch.zip
        if not os.path.exists(root):
            download_path = Path(root).parent / "temp_downloads"
            os.makedirs(download_path, exist_ok=True)
            print(f"Downloading imagenet_sketch in {download_path} and moving to {root}")
            old_cwd = os.getcwd()
            os.chdir(download_path)
            call(
                "wget https://huggingface.co/datasets/songweig/imagenet_sketch/resolve/main/data/ImageNet-Sketch.zip",
                shell=True,
            )
            print("Extracting imagenet_sketch (zip)")
            call("unzip -n ImageNet-Sketch.zip", shell=True)
            call(f"mv sketch imagenet_sketch", shell=True)
            call(f"mv imagenet_sketch {root}", shell=True)
            os.chdir(old_cwd)
        ds = ImageFolder(root=root, transform=transform, **kwargs)
        ds.classes = default_classnames["imagenet1k"]
    elif dataset_name == "imagenet-a":
        assert split == "test", f"Only `test` split available for {dataset_name}"
        # Downloadable from https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
        if not os.path.exists(root):
            download_path = Path(root).parent / "temp_downloads"
            os.makedirs(download_path, exist_ok=True)
            print(f"Downloading imagenet-a in {download_path} and moving to {root}")
            old_cwd = os.getcwd()
            os.chdir(download_path)
            call("wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar", shell=True)
            print(f"Extracting imagenet-a")
            call("tar xf imagenet-a.tar", shell=True)
            call(f"mv imagenet-a {root}", shell=True)
            os.chdir(old_cwd)
        ds = ImageFolder(root=root, transform=transform, **kwargs)
        ds.classes = default_classnames["imagenet1k"]
        imagenet_a_mask = [wnid in set(IMAGENET_A_WNIDS) for wnid in ALL_IMAGENET_WORDNET_IDS]
        ds.classes = [cl for cl, mask in zip(ds.classes, imagenet_a_mask) if mask]
    elif dataset_name == "imagenet-r":
        assert split == "test", f"Only `test` split available for {dataset_name}"
        # downloadable from https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
        if not os.path.exists(root):
            download_path = Path(root).parent / "temp_downloads"
            os.makedirs(download_path, exist_ok=True)
            print(f"Downloading imagenet-r in {download_path} and moving to {root}")
            old_cwd = os.getcwd()
            os.chdir(download_path)
            call("wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar", shell=True)
            print(f"Extracting imagenet-r")
            call("tar xf imagenet-r.tar", shell=True)
            call(f"mv imagenet-r {root}", shell=True)
            os.chdir(old_cwd)
        imagenet_r_mask = [wnid in IMAGENET_R_WNIDS for wnid in ALL_IMAGENET_WORDNET_IDS]
        ds = ImageFolder(root=root, transform=transform, **kwargs)
        ds.classes = default_classnames["imagenet1k"]
        ds.classes = [cl for cl, mask in zip(ds.classes, imagenet_r_mask) if mask]
    elif dataset_name == "imagenet-o":
        assert split == "test", f"Only `test` split available for {dataset_name}"
        # downloadable from https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar
        if not os.path.exists(root):
            download_path = Path(root).parent / "temp_downloads"
            os.makedirs(download_path, exist_ok=True)
            print(f"Downloading imagenet-o in {download_path} and moving to {root}")
            old_cwd = os.getcwd()
            os.chdir(download_path)
            call("wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar", shell=True)
            print(f"Extracting imagenet-o")
            call("tar xf imagenet-o.tar", shell=True)
            call(f"mv imagenet-o {root}", shell=True)
            os.chdir(old_cwd)
        ds = ImageFolder(root=root, transform=transform, **kwargs)
        ds.classes = default_classnames["imagenet1k"]
        imagenet_o_mask = [wnid in set(IMAGENET_O_WNIDS) for wnid in ALL_IMAGENET_WORDNET_IDS]
        ds.classes = [cl for cl, mask in zip(ds.classes, imagenet_o_mask) if mask]
    elif dataset_name == "objectnet":
        assert split == "test", f"Only `test` split available for {dataset_name}"
        # downloadable from https://objectnet.dev/downloads/objectnet-1.0.zip or https://www.dropbox.com/s/raw/cxeztdtm16nzvuw/objectnet-1.0.zip
        if not os.path.exists(root):
            download_path = Path(root).parent / "temp_downloads"
            os.makedirs(download_path, exist_ok=True)
            print(f"Downloading objectnet in {download_path} and moving to {root}")
            old_cwd = os.getcwd()
            os.chdir(download_path)
            if not Path("objectnet-1.0.zip").exists():
                call("wget https://objectnet.dev/downloads/objectnet-1.0.zip", shell=True)
            print(f"Extracting objectnet")
            call(
                "UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -n -P objectnetisatestset objectnet-1.0.zip",
                shell=True,
            )
            os.makedirs(root)
            call(f"mv objectnet-1.0 {root}", shell=True)
            call(f"cp {root}/objectnet-1.0/mappings/* {root}", shell=True)
        ds = objectnet.ObjectNetDataset(root=root, transform=transform)
    elif dataset_name == "voc2007":
        assert split in (
            "train",
            "test",
        ), f"Only `train` and `test` split available for {dataset_name}"
        ds = voc2007.PASCALVoc2007Cropped(
            root=root, set=split, transform=transform, download=download, **kwargs
        )
    elif dataset_name == "voc2007_multilabel":
        assert split in (
            "train",
            "test",
        ), f"Only `train` and `test` split available for {dataset_name}"
        ds = voc2007.PASCALVoc2007(
            root=root, set=split, transform=transform, download=download, **kwargs
        )
    elif dataset_name.startswith("sugar_crepe"):
        # https://github.com/RAIVNLab/sugar-crepe/tree/main
        _, task = dataset_name.split("/")
        assert task in (
            "add_att",
            "add_obj",
            "replace_att",
            "replace_obj",
            "replace_rel",
            "swap_att",
            "swap_obj",
        ), f"Unknown task {task} for {dataset_name}"
        assert split == "test", f"Only `test` split available for {dataset_name}"
        archive_name = "val2017.zip"
        root_split = os.path.join(root, archive_name.replace(".zip", ""))
        if not os.path.exists(root_split):
            print(f"Downloading coco captions {archive_name}...")
            if not os.path.exists(os.path.join(root, archive_name)):
                call(
                    f"wget http://images.cocodataset.org/zips/{archive_name} --output-document={root}/{archive_name}",
                    shell=True,
                )
            call(f"unzip {root}/{archive_name} -d {root}", shell=True)
        ann = f"{root}/{task}.json"
        if not os.path.exists(ann):
            url = f"https://raw.githubusercontent.com/RAIVNLab/sugar-crepe/main/data/{task}.json"
            call(f"wget {url} --output-document={ann}", shell=True)
        ds = sugar_crepe.SugarCrepe(
            root=os.path.join(root, "val2017"), ann_file=ann, transform=transform, **kwargs
        )
    elif dataset_name == "winoground":
        ds = winoground.WinoGround(root=root, transform=transform)
    elif dataset_name == "mscoco_captions":
        print(f"Untested, use entitynet.datasets.coco instead")
        # https://github.com/mehdidc/retrieval_annotations/releases/tag/1.0.0(annotations)
        if split == "train":
            archive_name = "train2014.zip"
        elif split in ("val", "test"):
            archive_name = "val2014.zip"
        else:
            raise ValueError(f"split should be `train` or `val` or `test` for `{dataset_name}`")
        root_split = os.path.join(root, archive_name.replace(".zip", ""))
        if not os.path.exists(root_split):
            print(f"Downloading mscoco_captions {archive_name}...")
            if not os.path.exists(os.path.join(root, archive_name)):
                call(
                    f"wget http://images.cocodataset.org/zips/{archive_name} --output-document={root}/{archive_name}",
                    shell=True,
                )
            call(f"unzip {root}/{archive_name} -d {root}", shell=True)
        if not annotation_file:
            annotation_file = f"{root}/coco_{split}_karpathy.json"
        if not os.path.exists(annotation_file):
            call(
                f"wget https://github.com/mehdidc/retrieval_annotations/releases/download/1.0.0/coco_{split}_karpathy.json --output-document={annotation_file}",
                shell=True,
            )
        ds = CocoCaptions(root=root_split, annFile=annotation_file, transform=transform, **kwargs)
    elif dataset_name == "multilingual_mscoco_captions":
        from clip_benchmark.datasets import multilingual_mscoco

        if language not in multilingual_mscoco.SUPPORTED_LANGUAGES:
            raise ValueError("Unsupported language for multilingual_ms_coco:", language)

        annotation_file = os.path.join(
            root, multilingual_mscoco.OUTPUT_FILENAME_TEMPLATE.format(language)
        )
        if not os.path.exists(annotation_file):
            multilingual_mscoco.create_annotation_file(root, language)

        ds = multilingual_mscoco.Multilingual_MSCOCO(
            root=root, ann_file=annotation_file, transform=transform, **kwargs
        )
    elif dataset_name == "crossmodal3600":
        from clip_benchmark.datasets import crossmodal3600

        if language not in crossmodal3600.SUPPORTED_LANGUAGES:
            raise ValueError("Unsupported language for Crossmodal-3600:", language)

        annotation_file = os.path.join(
            root, crossmodal3600.OUTPUT_FILENAME_TEMPLATE.format(language)
        )
        if not os.path.exists(annotation_file):
            crossmodal3600.create_annotation_file(root, language)

        ds = crossmodal3600.Crossmodal3600(
            root=root, ann_file=annotation_file, transform=transform, **kwargs
        )
    elif dataset_name == "xtd10":
        from clip_benchmark.datasets import xtd10

        if language not in xtd10.SUPPORTED_LANGUAGES:
            raise ValueError("Unsupported language for xtd10:", language)

        annotation_file = os.path.join(root, xtd10.OUTPUT_FILENAME_TEMPLATE.format(language))
        if not os.path.exists(annotation_file):
            xtd10.create_annotation_file(root, language)

        ds = xtd10.XTD10(root=root, ann_file=annotation_file, transform=transform, **kwargs)
    elif dataset_name == "xtd200":
        from clip_benchmark.datasets import xtd200

        if language not in xtd200.SUPPORTED_LANGUAGES:
            raise ValueError("Unsupported language for xtd200:", language)

        annotation_file = os.path.join(root, xtd200.OUTPUT_FILENAME_TEMPLATE.format(language))
        if not os.path.exists(annotation_file):
            xtd200.create_annotation_file(root, language)

        ds = xtd200.XTD200(root=root, ann_file=annotation_file, transform=transform, **kwargs)
    elif dataset_name == "flickr30k-200":
        from clip_benchmark.datasets import flickr30k_200

        if language not in flickr30k_200.SUPPORTED_LANGUAGES:
            raise ValueError("Unsupported language for flickr30k-200:", language)

        annotation_file = os.path.join(
            root, flickr30k_200.OUTPUT_FILENAME_TEMPLATE.format(language)
        )
        if not os.path.exists(annotation_file):
            flickr30k_200.create_annotation_file(root, language)

        ds = flickr30k_200.Flickr30k_200(
            root=root, ann_file=annotation_file, transform=transform, **kwargs
        )
    elif dataset_name == "flickr30k":
        # downloadable from https://www.kaggle.com/datasets/adityajn105/flickr30k
        # https://github.com/mehdidc/retrieval_annotations/releases/tag/1.0.0 (annotations)
        # `kaggle datasets download -d adityajn105/flickr30k`
        assert split in (
            "train",
            "val",
            "test",
        ), f"Only `train` and `val` and `test` split available for {dataset_name}"
        if not os.path.exists(root):
            download_path = Path(root).parent / "temp_downloads"
            os.makedirs(download_path, exist_ok=True)
            print(f"Downloading flickr30k in {download_path} and moving to {root}")
            old_cwd = os.getcwd()
            os.chdir(download_path)
            call(
                "wget https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images.zip",
                shell=True,
            )
            print(f"Extracting flickr30k")
            call("unzip -qn flickr30k-images.zip", shell=True)
            os.makedirs(root, exist_ok=True)
            call(f"mv flickr30k-images {root}/images", shell=True)
            os.chdir(old_cwd)
        # # disabled kaggle because it requires login
        # if not os.path.exists(root):
        #     # Automatic download
        #     print(f"Downloading flickr30k to {root}")
        #     if not has_kaggle():
        #         raise RuntimeError(
        #             "Kaggle is needed to download the dataset. Please install it via `pip install kaggle`"
        #         )
        #     call("kaggle datasets download -d hsankesara/flickr-image-dataset", shell=True)
        #     call(f"unzip flickr-image-dataset.zip", shell=True)
        #     call(
        #         f"mv flickr30k_images/flickr30k_images {root} && rm -rf flickr30k_images",
        #         shell=True,
        #     )
        if not annotation_file:
            if language == "en":
                annotation_file = f"{root}/flickr30k_{split}_karpathy.txt"
            elif language == "zh":
                annotation_file = f"{root}/flickr30k_{split}_zh.txt"
            else:
                raise ValueError(f"Unsupported language {language} for `{dataset_name}`")
        if not os.path.exists(annotation_file):
            # Download Flickr30K Karpathy test set
            if language == "en":
                call(
                    f"wget https://github.com/mehdidc/retrieval_annotations/releases/download/1.0.0/flickr30k_{split}_karpathy.txt --output-document={annotation_file}",
                    shell=True,
                )
            elif language == "zh":
                call(
                    f"wget https://github.com/mehdidc/retrieval_annotations/releases/download/1.0.0/flickr30k_{split}_zh.txt --output-document={annotation_file}",
                    shell=True,
                )
            else:
                raise ValueError(f"Unsupported language {language} for `{dataset_name}`")
        ds = flickr.Flickr(root=root, ann_file=annotation_file, transform=transform, **kwargs)
    elif dataset_name == "flickr8k":
        assert split in (
            "train",
            "val",
            "test",
        ), f"Only `train` and `val` and `test` split available for {dataset_name}"
        # downloadable from https://www.kaggle.com/datasets/adityajn105/flickr8k
        # `kaggle datasets download -d adityajn105/flickr8k`
        # https://github.com/mehdidc/retrieval_annotations/releases/tag/1.0.0(annotations)
        if not os.path.exists(root):
            # Automatic download
            print(f"Downloading flickr8k to {root}")
            if not has_kaggle():
                raise RuntimeError(
                    "Kaggle is needed to download the dataset. Please install it via `pip install kaggle`"
                )
            call("kaggle datasets download -d adityajn105/flickr8k", shell=True)
            call(f"unzip flickr8k.zip", shell=True)
            call(f"mv Images {root}", shell=True)
            call(f"mv captions.txt {root}", shell=True)
        if not annotation_file:
            if language == "en":
                annotation_file = f"{root}/flickr8k_{split}_karpathy.txt"
            elif language == "zh":
                annotation_file = f"{root}/flickr8k_{split}_zh.txt"
            else:
                raise ValueError(f"Unsupported language {language} for `{dataset_name}`")
        if not os.path.exists(annotation_file):
            # Download Flickr8K Karpathy test set
            if language == "en":
                call(
                    f"wget https://github.com/mehdidc/retrieval_annotations/releases/download/1.0.0/flickr8k_{split}_karpathy.txt --output-document={annotation_file}",
                    shell=True,
                )
            elif language == "zh":
                call(
                    f"wget https://github.com/mehdidc/retrieval_annotations/releases/download/1.0.0/flickr8k_{split}_zh.txt --output-document={annotation_file}",
                    shell=True,
                )
            else:
                raise ValueError(f"Unsupported language {language} for `{dataset_name}`")
        ds = flickr.Flickr(root=root, ann_file=annotation_file, transform=transform, **kwargs)
    elif dataset_name == "food101":
        assert split in (
            "train",
            "test",
        ), f"Only `train` and `test` split available for {dataset_name}"
        ds = Food101(root=root, split=split, transform=transform, download=download, **kwargs)
        # we use the default class names, we just  replace "_" by spaces
        # to delimit words
        ds.classes = [cl.replace("_", " ") for cl in ds.classes]
    elif dataset_name == "sun397":
        assert split == "test", f"Only `test` split available for {dataset_name} but got {split}"
        # we use the default class names, we just  replace "_" and "/" by spaces to delimit words
        ds = SUN397CachedPaths(root=root, transform=transform, download=download, **kwargs)
        ds.classes = [cl.replace("_", " ").replace("/", " ") for cl in ds.classes]
    elif dataset_name == "cars":
        assert split in (
            "train",
            "test",
        ), f"Only `train` and `test` split available for {dataset_name}"
        logger.info(f"Loading cars in {root}")
        ds = StanfordCars(root=root, split=split, transform=transform, download=False, **kwargs)
    elif dataset_name == "fgvc_aircraft":
        assert split in (
            "train",
            "val",
            "trainval",
            "test",
        ), f"Only `train` and `val` and `trainval` and `test` split available for {dataset_name}"
        ds = FGVCAircraft(
            root=root,
            annotation_level="variant",
            split=split,
            transform=transform,
            download=download,
            **kwargs,
        )
    elif dataset_name == "dtd":
        assert split in (
            "train",
            "val",
            "test",
        ), f"Only `train` and `val` and `test` split available for {dataset_name}"
        ds = DTD(root=root, split=split, transform=transform, download=download, **kwargs)
    elif dataset_name == "pets":
        assert split in (
            "trainval",
            "test",
        ), f"Only `trainval` and `test` split available for {dataset_name}"
        ds = OxfordIIITPet(
            root=root,
            split=split,
            target_types="category",
            transform=transform,
            download=download,
            **kwargs,
        )
    elif dataset_name == "caltech101":
        warnings.warn(
            f"split argument ignored for `{dataset_name}`, there are no pre-defined train/test splits for this dataset"
        )
        # broken download link (can't download google drive), fixed by this PR https://github.com/pytorch/vision/pull/5645
        # also available in "vtab/caltech101" using VTAB splits, we advice to use VTAB version rather than this one
        # since in this one (torchvision) there are no pre-defined test splits
        ds = caltech101.Caltech101(
            root=root, target_type="category", transform=transform, download=download, **kwargs
        )
        ds.classes = default_classnames["caltech101"]
    elif dataset_name == "flowers":
        assert split in (
            "train",
            "val",
            "test",
        ), f"Only `train` and `val` and `test` split available for {dataset_name}"
        ds = Flowers102(root=root, split=split, transform=transform, download=download, **kwargs)
        # class indices started by 1 until it was fixed in  a  PR (#TODO link of the PR)
        # if older torchvision version, fix it using a target transform that decrements label index
        # TODO figure out minimal torchvision version needed instead of decrementing
        if ds[0][1] == 1:
            ds.target_transform = lambda y: y - 1
        ds.classes = default_classnames["flowers"]
    elif dataset_name == "mnist":
        assert split in (
            "train",
            "test",
        ), f"Only `train` and `test` split available for {dataset_name}"
        ds = MNIST(root=root, train=train, transform=transform, download=download, **kwargs)
        ds.classes = default_classnames["mnist"]
    elif dataset_name == "stl10":
        assert split in (
            "train",
            "test",
        ), f"Only `train` and `test` split available for {dataset_name}"
        ds = STL10(root=root, split=split, transform=transform, download=download, **kwargs)
    elif dataset_name == "eurosat":
        warnings.warn(
            f"split argument ignored for `{dataset_name}`, there are no pre-defined train/test splits for this dataset"
        )
        ds = EuroSAT(root=root, transform=transform, download=download, **kwargs)
        ds.classes = default_classnames["eurosat"]
    elif dataset_name == "gtsrb":
        assert split in (
            "train",
            "test",
        ), f"Only `train` and `test` split available for {dataset_name}"
        ds = GTSRB(root=root, split=split, transform=transform, download=download, **kwargs)
        ds.classes = default_classnames["gtsrb"]
    elif dataset_name == "country211":
        assert split in (
            "train",
            "valid",
            "test",
        ), f"Only `train` and `valid` and `test` split available for {dataset_name}"
        print(f"init country211 in {root}")
        ds = Country211(root=root, split=split, transform=transform, download=download, **kwargs)
        ds.classes = default_classnames["country211"]
    elif dataset_name == "pcam":
        assert split in (
            "train",
            "val",
            "test",
        ), f"Only `train` and `val` and `test` split available for {dataset_name}"
        # Dead link. Fixed by this PR on torchvision https://github.com/pytorch/vision/pull/5645
        # TODO figure out minimal torchvision version needed
        ds = PCAM(root=root, split=split, transform=transform, download=download, **kwargs)
        ds.classes = default_classnames["pcam"]
    elif dataset_name == "renderedsst2":
        assert split in (
            "train",
            "val",
            "test",
        ), f"Only `train` and `val` and `test` split available for {dataset_name}"
        ds = RenderedSST2(root=root, split=split, transform=transform, download=download, **kwargs)
    elif dataset_name == "fer2013":
        assert split in (
            "train",
            "test",
        ), f"Only `train` and `test` split available for {dataset_name}"
        # Downloadable from  https://www.kaggle.com/datasets/msambare/fer2013
        # `kaggle datasets download -d msambare/fer2013`
        if not os.path.exists(root):
            # Automatic download
            print("Downloading fer2013...")
            if not has_kaggle():
                raise RuntimeError(
                    "Kaggle is needed to download the dataset. Please install it via `pip install kaggle`"
                )
            call("kaggle datasets download -d msambare/fer2013", shell=True)
            call(f"unzip fer2013.zip -d {root}", shell=True)
        root = os.path.join(root, "train" if train else "test")
        ds = ImageFolder(root=root, transform=transform)
        ds.classes = default_classnames["fer2013"]
    elif dataset_name.startswith("tfds/"):
        # TFDS datasets support using `timm` and `tensorflow_datasets`
        prefix, *name_list = dataset_name.split("/")
        name = "/".join(name_list)
        ds = build_tfds_dataset(
            name, download=download, split=split, data_dir=root, transform=transform
        )
    elif dataset_name.startswith("vtab/"):
        # VTAB datasets support using `tensorflow_datasets` and `task_adaptation`
        prefix, *name_list = dataset_name.split("/")
        name = "/".join(name_list)
        ds = build_vtab_dataset(
            name,
            download=download,
            split=split,
            data_dir=root,
            transform=transform,
            classnames=default_classnames,
        )
    elif dataset_name.startswith("wds/"):
        # WebDataset support using `webdataset` library
        name = dataset_name.split("/", 1)[1]
        ds = build_wds_dataset(
            name, transform=transform, split=split, data_dir=root, cache_dir=wds_cache_dir
        )
        # WDS specify classnames and templates on its own.
    elif dataset_name == "dummy":
        ds = Dummy()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}.")

    default_dataset_for_templates = "imagenet1k"
    if (
        dataset_name.startswith("tfds/")
        or dataset_name.startswith("vtab/")
        or dataset_name.startswith("wds/")
    ):
        prefix, *rest = dataset_name.split("/")
        short_name = "/".join(rest)
        # if it's a vtab/tfds/wds/ dataset, we look for e.g. vtab/<name>
        # as well as <name> in the custom template file/classname file,
        # whichever is found.
        keys_to_lookup = [dataset_name, short_name]
    else:
        keys_to_lookup = [dataset_name]

    if use_classnames_and_templates:
        # Specify templates for the dataset (if needed)
        if custom_templates:
            # We override with custom templates ONLY if they are provided,
            # which is the case when `custom_templates` is loaded.
            ds.templates = value_from_first_key_found(
                custom_templates, keys=keys_to_lookup + [default_dataset_for_templates]
            )
            assert ds.templates is not None, f"Templates not specified for {dataset_name}"
        elif not hasattr(ds, "templates"):
            # No templates specified by the dataset itself,
            # so we use  templates are packaged with CLIP benchmark
            # (loaded from <LANG>_zeroshot_classification_templates.json).
            ds.templates = value_from_first_key_found(
                default_templates, keys=keys_to_lookup + [default_dataset_for_templates]
            )
            assert ds.templates is not None, f"Templates not specified for {dataset_name}"
        else:
            # dataset has templates already (e.g., WDS case), so we keep it as is.
            pass

        # We override with custom classnames ONLY if they are provided.
        if custom_classnames:
            ds.classes = value_from_first_key_found(custom_classnames, keys=keys_to_lookup)

        assert ds.classes is not None, f"Classes not specified for {dataset_name}"
        assert ds.templates is not None, f"Templates not specified for {dataset_name}"
    return ds


def value_from_first_key_found(dic, keys):
    for k in keys:
        if k in dic:
            return dic[k]


class Dummy:

    def __init__(self):
        self.classes = ["blank image", "noisy image"]

    def __getitem__(self, i):
        return torch.zeros(3, 224, 224), 0

    def __len__(self):
        return 1


def get_dataset_default_task(dataset):
    if dataset in (
        "flickr30k",
        "flickr8k",
        "mscoco_captions",
        "multilingual_mscoco_captions",
        "flickr30k-200",
        "crossmodal3600",
        "xtd200",
    ):
        return "zeroshot_retrieval"
    elif dataset.startswith("sugar_crepe") or dataset == "winoground":
        return "image_caption_selection"
    else:
        return "zeroshot_classification"


def get_dataset_collate_fn(dataset_name):
    if dataset_name in (
        "mscoco_captions",
        "multilingual_mscoco_captions",
        "flickr30k",
        "flickr8k",
        "flickr30k-200",
        "crossmodal3600",
        "xtd200",
        "winoground",
    ) or dataset_name.startswith("sugar_crepe"):
        return image_captions_collate_fn
    else:
        return default_collate


def has_gdown():
    return call("which gdown", shell=True) == 0


def has_kaggle():
    return call("which kaggle", shell=True) == 0


def build_vtab_dataset(
    dataset_name, transform, download=True, split="test", data_dir="root", classnames=None
):
    # Using VTAB splits instead of default TFDS splits
    if classnames is None:
        classnames = []
    from .tfds import VTABIterableDataset, disable_gpus_on_tensorflow, download_tfds_dataset

    # avoid Tensorflow owning GPUs to not clash with PyTorch
    disable_gpus_on_tensorflow()

    # by default we take classes from TFDS (default behavior if `classes` stays None),
    # except for the datasets that will override `classes` (e.g., clevr_*)
    classes = None
    if dataset_name == "caltech101":
        from task_adaptation.data.caltech import Caltech101

        tfds_dataset = Caltech101(data_dir=data_dir)
        classes = classnames["caltech101_vtab"]
    elif dataset_name == "cars":
        raise NotImplementedError(
            f"use clip_benchmark 'cars' instead. 'vtab/cars' is dead since the author deleted it "
            f"from the homepage: https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz"
        )
        # from task_adaptation.data.cars import CarsData
        # tfds_dataset = CarsData(data_dir=data_dir)
    elif dataset_name in ("cifar10", "cifar100"):
        from task_adaptation.data.cifar import CifarData

        tfds_dataset = CifarData(
            data_dir=data_dir, num_classes=10 if dataset_name == "cifar10" else 100
        )
    elif dataset_name.startswith("clevr_"):
        from task_adaptation.data.clevr import CLEVRData

        task = _extract_task(dataset_name)
        assert task in ("count_all", "closest_object_distance")
        tfds_dataset = CLEVRData(task=task, data_dir=data_dir)
        if task == "count_all":
            classes = classnames["clevr_count_all"]
        elif task == "closest_object_distance":
            classes = classnames["clevr_closest_object_distance"]
        else:
            raise ValueError(f"non supported: {task}")
    elif dataset_name == "cub":
        from task_adaptation.data.cub import CUB2011Data

        tfds_dataset = CUB2011Data(data_dir=data_dir)
    elif dataset_name == "diabetic_retinopathy":
        # Needs manual download from Kaggle
        # 1) `kaggle competitions download -c diabetic-retinopathy-detection` on $ROOT/downloads/manual
        # 2) extract archives  on $ROOT/downloads/manual
        if not os.path.exists(data_dir):
            # Automatic download
            print(f"Downloading diabetic_retinopathy to {data_dir}")
            if not has_kaggle():
                raise RuntimeError(
                    "Kaggle is needed to download the dataset. Please install it via `pip install kaggle`"
                )
            os.makedirs(os.path.join(data_dir, "downloads", "manual"))
            call(
                f"kaggle competitions download -c diabetic-retinopathy-detection -p {data_dir}/downloads/manual",
                shell=True,
            )
            call(
                f"cd {data_dir}/downloads/manual;unzip diabetic-retinopathy-detection.zip;cat train.zip*>train.zip;cat test.zip*>test.zip;unzip train.zip; unzip test.zip;unzip sample.zip;unzip trainLabels.csv.zip",
                shell=True,
            )
        from task_adaptation.data.diabetic_retinopathy import RetinopathyData

        tfds_dataset = RetinopathyData(config="btgraham-300", data_dir=data_dir)
        classes = classnames["diabetic_retinopathy"]
    elif dataset_name == "dmlab":
        from task_adaptation.data.dmlab import DmlabData

        download_tfds_dataset(
            "dmlab", data_dir=data_dir
        )  # it's not called in the original VTAB code, so we do it explictly
        tfds_dataset = DmlabData(data_dir=data_dir)
        classes = classnames["dmlab"]
    elif dataset_name.startswith("dsprites_"):
        from task_adaptation.data.dsprites import DSpritesData

        task = _extract_task(dataset_name)
        assert task in (
            "label_shape",
            "label_scale",
            "label_orientation",
            "label_x_position",
            "label_y_position",
        )
        tfds_dataset = DSpritesData(task, data_dir=data_dir)
        classes = tfds_dataset._dataset_builder.info.features[task].names
    elif dataset_name == "dtd":
        from task_adaptation.data.dtd import DTDData

        tfds_dataset = DTDData(data_dir=data_dir)
    elif dataset_name == "eurosat":
        from task_adaptation.data.eurosat import EurosatData

        tfds_dataset = EurosatData(subset="rgb", data_key="image", data_dir=data_dir)
        classes = classnames["eurosat"]
    elif dataset_name == "food101":
        from task_adaptation.data.food101 import Food101Data

        tfds_dataset = Food101Data(data_dir=data_dir)
    elif dataset_name == "inaturalist":
        from task_adaptation.data.inaturalist import INaturalistData

        tfds_dataset = INaturalistData(data_dir=data_dir, year=2017)
    elif dataset_name.startswith("kitti_"):
        from .kitti import KittiData

        task = _extract_task(dataset_name)
        assert task in (
            "count_all",
            "count_left",
            "count_far",
            "count_near",
            "closest_object_distance",
            "closest_object_x_location",
            "count_vehicles",
            "closest_vehicle_distance",
        )
        tfds_dataset = KittiData(task=task, data_dir=data_dir)
        if task == "closest_vehicle_distance":
            classes = classnames["kitti_closest_vehicle_distance"]
        else:
            raise ValueError(f"Unsupported task: {task=} only closest_vehicle_distance is supp.")
    elif dataset_name == "flowers":
        from task_adaptation.data.oxford_flowers102 import OxfordFlowers102Data

        tfds_dataset = OxfordFlowers102Data(data_dir=data_dir)
    elif dataset_name == "pets":
        from task_adaptation.data.oxford_iiit_pet import OxfordIIITPetData

        tfds_dataset = OxfordIIITPetData(data_dir=data_dir)
        classes = classnames["pets"]
    elif dataset_name == "pcam":
        from task_adaptation.data.patch_camelyon import PatchCamelyonData

        tfds_dataset = PatchCamelyonData(data_dir=data_dir)
        classes = classnames["pcam"]
    elif dataset_name == "resisc45":
        # Needs download from OneDrive: https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs
        # The archive needs to to be put at <DATASET_ROOT>/downloads/manual then extracted
        if not os.path.exists(data_dir):
            os.makedirs(os.path.join(data_dir, "downloads", "manual"))
            call(
                f"wget 'https://onedrive.live.com/download?resid=5C5E061130630A68!107&authkey=!AHHNaHIlzp_IXjs' --output-document={data_dir}/downloads/manual/resisc45.rar",
                shell=True,
            )
            call(f"cd {data_dir}/downloads/manual;unrar x resisc45.rar", shell=True)
        from task_adaptation.data.resisc45 import Resisc45Data

        tfds_dataset = Resisc45Data(data_dir=data_dir)
    elif dataset_name.startswith("smallnorb_"):
        from task_adaptation.data.smallnorb import SmallNORBData

        task = _extract_task(dataset_name)
        assert task in ("label_category", "label_elevation", "label_azimuth", "label_lighting")
        tfds_dataset = SmallNORBData(predicted_attribute=task, data_dir=data_dir)
        classes = tfds_dataset._dataset_builder.info.features[task].names
    elif dataset_name == "sun397":
        from task_adaptation.data.sun397 import Sun397Data

        # FIXME There is a problem in `sun397`, when TFDS tries download it
        # there is an image that cannot be decoded. For the time being
        # we will use torchvision's SUN397 instead.
        tfds_dataset = Sun397Data(config="tfds", data_dir=data_dir)
    elif dataset_name == "svhn":
        from task_adaptation.data.svhn import SvhnData

        tfds_dataset = SvhnData(data_dir=data_dir)
        classes = classnames["svhn"]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    ds = VTABIterableDataset(
        tfds_dataset,
        input_name="image",
        label_name="label",
        transform=transform,
        target_transform=int,
        split=split,
        classes=classes,
    )
    return ds


def build_tfds_dataset(name, transform, download=True, split="test", data_dir="root", classes=None):
    from .tfds import disable_gpus_on_tensorflow

    disable_gpus_on_tensorflow()
    import tensorflow_datasets as tfds
    import timm

    builder = tfds.builder(name, data_dir=data_dir)
    if download:
        builder.download_and_prepare()
    splits = list(builder.info.splits.keys())
    assert split in splits, (split, splits)
    ds = timm.data.create_dataset(
        f"tfds/{name}", data_dir, split=split, transform=transform, target_transform=int
    )
    ds.classes = builder.info.features["label"].names if classes is None else classes
    return ds


def build_wds_dataset(dataset_name, transform, split="test", data_dir="root", cache_dir=None):
    """
    Load a dataset in WebDataset format. Either local paths or HTTP URLs can be specified.
    Expected file structure is:
    ```
    data_dir/
        train/
            nshards.txt
            0.tar
            1.tar
            ...
        test/
            nshards.txt
            0.tar
            1.tar
            ...
        classnames.txt
        zeroshot_classification_templates.txt
        dataset_type.txt
    ```
    Classnames and templates are required for zeroshot classification, while dataset type
    (equal to "retrieval") is required for zeroshot retrieval datasets.

    You can use the `clip_benchmark_export_wds` or corresponding API
    (`clip_benchmark.webdataset_builder.convert_dataset`) to convert datasets to this format.

    Set `cache_dir` to a path to cache the dataset, otherwise, no caching will occur.
    """
    import webdataset as wds

    def read_txt(fname):
        if "://" in fname:
            stream = os.popen("curl -L -s --fail '%s'" % fname, "r")
            value = stream.read()
            if stream.close():
                raise FileNotFoundError("Failed to retreive data")
        else:
            with open(fname, "r") as file:
                value = file.read()
        return value

    # Special handling for Huggingface datasets
    # Git LFS files have a different file path to access the raw data than other files
    if data_dir.startswith("https://huggingface.co/datasets"):
        # Format: https://huggingface.co/datasets/<USERNAME>/<REPO>/tree/<BRANCH>
        *split_url_head, _, url_path = data_dir.split("/", 7)
        url_head = "/".join(split_url_head)
        metadata_dir = "/".join([url_head, "raw", url_path])
        tardata_dir = "/".join([url_head, "resolve", url_path])
    else:
        metadata_dir = tardata_dir = data_dir
    # Get number of shards
    nshards_fname = os.path.join(metadata_dir, split, "nshards.txt")
    nshards = int(
        read_txt(nshards_fname)
    )  # Do not catch FileNotFound, nshards.txt should be mandatory
    # Get dataset type (classification or retrieval)
    type_fname = os.path.join(metadata_dir, "dataset_type.txt")
    try:
        dataset_type = read_txt(type_fname).strip().lower()
    except FileNotFoundError:
        # print("WARNING: dataset_type.txt not found, assuming type=classification")
        dataset_type = "classification"
    #
    filepattern = os.path.join(tardata_dir, split, "{0..%d}.tar" % (nshards - 1))
    # Load webdataset (support WEBP, PNG, and JPG for now)
    if not cache_dir or not isinstance(cache_dir, str):
        cache_dir = None
    dataset = wds.WebDataset(filepattern, cache_dir=cache_dir, nodesplitter=lambda src: src).decode(
        wds.autodecode.ImageHandler("pil", extensions=["webp", "png", "jpg", "jpeg"])
    )
    # Load based on classification or retrieval task
    if dataset_type == "retrieval":
        dataset = dataset.to_tuple(["webp", "png", "jpg", "jpeg"], "txt").map_tuple(
            transform, str.splitlines
        )
        dataset.classes = dataset.templates = None
    else:
        label_type = "npy" if dataset_type == "multilabel" else "cls"  # Special case for multilabel
        dataset = dataset.to_tuple(["webp", "png", "jpg", "jpeg"], label_type).map_tuple(
            transform, None
        )
        # Get class names if present
        classnames_fname = os.path.join(metadata_dir, "classnames.txt")
        try:
            dataset.classes = [line.strip() for line in read_txt(classnames_fname).splitlines()]
        except FileNotFoundError:
            print("WARNING: classnames.txt not found")
            dataset.classes = None
        # Get zeroshot classification templates if present
        templates_fname = os.path.join(metadata_dir, "zeroshot_classification_templates.txt")
        try:
            dataset.templates = [line.strip() for line in read_txt(templates_fname).splitlines()]
        except FileNotFoundError:
            print("WARNING: zeroshot_classification_templates.txt not found")
            dataset.templates = None

    return dataset


def _extract_task(dataset_name):
    prefix, *task_name_list = dataset_name.split("_")
    task = "_".join(task_name_list)
    return task


def image_captions_collate_fn(batch):
    transposed = list(zip(*batch))
    imgs = default_collate(transposed[0])
    texts = transposed[1]
    return imgs, texts


def get_dataset_collection_from_file(path):
    return [l.strip() for l in open(path).readlines()]


dataset_collection = {
    "vtab": [
        "vtab/caltech101",
        "vtab/cifar100",
        "vtab/clevr_count_all",
        "vtab/clevr_closest_object_distance",
        "vtab/diabetic_retinopathy",
        "vtab/dmlab",
        "vtab/dsprites_label_orientation",
        "vtab/dsprites_label_x_position",
        "vtab/dtd",
        "vtab/eurosat",
        "vtab/kitti_closest_vehicle_distance",
        "vtab/flowers",
        "vtab/pets",
        "vtab/pcam",
        "vtab/resisc45",
        "vtab/smallnorb_label_azimuth",
        "vtab/smallnorb_label_elevation",
        "sun397",
        "vtab/svhn",
    ],
    "vtab+": [
        "imagenet1k",
        "imagenetv2",
        "imagenet_sketch",
        "imagenet-a",
        "imagenet-r",
        "objectnet",
        "fer2013",
        "voc2007",
        "voc2007_multilabel",
        "sun397",
        "cars",
        "fgvc_aircraft",
        "mnist",
        "stl10",
        "gtsrb",
        "country211",
        "renderedsst2",
        "vtab/caltech101",
        "vtab/cifar10",
        "vtab/cifar100",
        "vtab/clevr_count_all",
        "vtab/clevr_closest_object_distance",
        "vtab/diabetic_retinopathy",
        "vtab/dmlab",
        "vtab/dsprites_label_orientation",
        "vtab/dsprites_label_x_position",
        "vtab/dtd",
        "vtab/eurosat",
        "vtab/kitti_closest_vehicle_distance",
        "vtab/flowers",
        "vtab/pets",
        "vtab/pcam",
        "vtab/resisc45",
        "vtab/smallnorb_label_azimuth",
        "vtab/smallnorb_label_elevation",
        "vtab/svhn",
    ],
    "retrieval": [
        "mscoco_captions",
        "flickr8k",
        "flickr30k",
    ],
    "imagenet_robustness": [
        "imagenetv2",
        "imagenet_sketch",
        "imagenet-a",
        "imagenet-r",
        "objectnet",
    ],
    "sugar_crepe": [
        "sugar_crepe/add_att",
        "sugar_crepe/add_obj",
        "sugar_crepe/replace_att",
        "sugar_crepe/replace_obj",
        "sugar_crepe/replace_rel",
        "sugar_crepe/swap_att",
        "sugar_crepe/swap_obj",
    ],
}
