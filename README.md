<div align="center">
<img src='https://raw.githubusercontent.com/lmb-freiburg/entitynet/refs/heads/main/assets/logo_cv_small.png' alt='Computer Vision, University of Freiburg' title='Computer Vision, University of Freiburg'/><br/>
<a href="https://arxiv.org/pdf/2505.02746">paper</a> —
<a href="https://arxiv.org/abs/2505.02746">arXiv</a> —
<a href="https://entity-net.github.io">project page</a> —
<a href="https://huggingface.co/collections/lmb-freiburg/entitynet-6810b98ea9288fef9b6c09ca">🤗 models</a> —
<a href="https://huggingface.co/datasets/lmb-freiburg/entitynet">🤗 dataset</a>
</div>

# EntityNet

_Using Knowledge Graphs to harvest datasets for efficient CLIP model training_

We use knowledge graphs and web image search to build a diverse dataset of 33M images paired with 46M texts.
We show that this dataset can be used to train a generic CLIP model in a short amount of time.

Using a 10M-image subset focused on living organisms, we train domain expert models that excel
at fine-grained classification of animals, plants, and fungi.

Stay tuned for the dataset release.

## News

- 19 August 2025: Added **evaluation code**
- 06 May 2025: **Models** on [Hugging Face](https://huggingface.co/collections/lmb-freiburg/entitynet-6810b98ea9288fef9b6c09ca)
- 05 May 2025: **Preprint** on [arXiv](https://arxiv.org/abs/2505.02746)
- 23 April 2025: **Talk** at [Stuttgart AI, Machine Learning and Computer Vision Meetup](https://voxel51.com/computer-vision-events/stuttgart-ai-ml-and-computer-vision-meetup-april-23-2025/)
- 20 March 2025: **Poster** at [2025 ELLIS Winter School on Foundation Models](https://ivi.fnwi.uva.nl/ellis/events/2025-ellis-winter-school-on-foundation-models-fomo/)

## Release roadmap

- [x] Publish preprint
- [x] Upload CLIP models on huggingface
- [x] Add evaluation code
- [ ] Add training dataset
  - [X] Add code to download images from URLs and create webdataset
  - [ ] Add dataloader for the webdataset
- [ ] Add model training code

## Models

Our CLIP models are available on [🤗 Hugging Face](https://huggingface.co/collections/lmb-freiburg/entitynet-6810b98ea9288fef9b6c09ca)

Models are named as follows:

- Architecture
- Trained on EntityNet-33M (all images) or trained on LivingThings-10M (only trained on images of living organisms)
- Pretrained from scratch or finetuned

### Usage with open_clip

Models can be used with [open_clip](https://github.com/mlfoundations/open_clip) as follows:

```python
import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:lmb-freiburg/CLIP-ViT-B-16-EntityNet-33M")
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-16', context_length=32)

image = preprocess(Image.open("assets/images_cc0/rabbit.png")).unsqueeze(0)
texts = ["a dog", "a cat", "a rabbit"]
tokens = tokenizer(texts)

with torch.no_grad(), torch.autocast("cuda"):
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    logits = (model.logit_scale * image_features @ text_features.T)
    pred_class = logits.argmax(-1).item()

print(texts[pred_class])  # prints: a rabbit

```

## Repository structure

All commands should be run from the repository root.

```bash
configs/  yaml files for the experiments:
  configs/projects/group/name.yaml
src/
  clip_benchmark/  # benchmark datasets from https://github.com/LAION-AI/CLIP_benchmark
  entitynet/  # entitynet related code
  open_clip/  #  clip model, loss etc from https://github.com/mlfoundations/open_clip
```

## Setup

- Currently running with python=3.12 torch=2.6 cuda=12.4
- Setup the paths with environment variables `ENTITYNET_DATA_DIR` and `ENTITYNET_OUTPUT_DIR`

```bash
# 0. setup conda environment
conda update conda -n base -y

_ENV=entitynet
conda create -n entitynet python=3.12 -y
conda activate entitynet
pip install torch torchvision
pip install -U -r requirements.txt
pip install -e .

python -m entitynet.cli.print_paths
```

### Prepare evaluation datasets

- Put all data into the path defined by environment variable `ENTITYNET_DATA_DIR`
- Dataset building code is in `src/clip_benchmark/datasets/builder.py` and `src/entitynet/dataset_factory.py`

```bash
# ----- ImageNet ILSVRC2012 and robustness benchmarks
echo setup files for ImageNet ILSVRC2012 manually as follows.
# imagenet1k/
#   train/n########/*.JPEG
#   val/n########/*.JPEG

# run the tests for the evaluation tasks, this will download the remaining datasets.
python -m pytest tests/entitynet/test_eval_tasks_imagenet.py -sx

# # resulting file structure
# clip_benchmark/imagenetv2/ImageNetV2-matched-frequency/*/*.jpeg
# clip_benchmark/imagenet-r/n********/*.jpg
# clip_benchmark/objectnet/
#   objectnet-1.0/images/<ClassName>/*.png
#   folder_to_objectnet_label.json
#   imagenet_to_label_2012_v2
#   objectnet_to_imagenet_1k.json
#   pytorch_to_imagenet_2012_id.json
# clip_benchmark/imagenet_sketch/n********/*.JPEG
# clip_benchmark/imagenet-a/n********/*.jpg

# ----- iNaturalist
./bash_scripts/dataset/setup_inaturalist.sh
# create the traindev splits
python -m entitynet.cli.datasets.inat19_generate_splits
python -m entitynet.cli.datasets.inat21_generate_splits

# # resulting file structure
# iNat/2019/
#   train_val2019/<Category>/<ClassId>/*.jpg
#   categories.json
#   train2019.json
#   val2019.json
#   inat2019_traindev_split.json
#   inat2019_trainnodev_split.json
# iNat/2021
#   train/<ClassId>_<FullName>/*.jpg
#   val/<ClassId>_<FullName>/*.jpg
#   train.json
#   val.json
#   inat2021_traindev_split.json
#   inat2021_trainnodev_split.json

python -m pytest tests/entitynet/test_eval_tasks_inat.py -sx

# ----- CUB
./bash_scripts/dataset/setup_cub.sh

# # resulting file structure
# cub200/CUB_200_2011/
#   images/<Nr>.<ClassName>/*.jpg
#   train_test_split.txt

python -m pytest tests/entitynet/test_eval_tasks_cub.py -sx

# ----- RareSpecies
# run test, it will setup the dataset
python -m pytest tests/entitynet/test_eval_tasks_rarespecies -sx

# # resulting file structure
# rare-species/
#   images/<FullName>/*.jpg

# ----- Coco
./bash_scripts/dataset/setup_coco.sh

# # resulting file structure
# coco/
#   splits_karpathy/
#     coco_karpathy_test.json
#     coco_karpathy_train.json
#     coco_karpathy_val.json

# run test
python -m pytest tests/entitynet/test_eval_tasks_coco.py -sx

# ----- Flickr30K
# run the test, it will setup the dataset
python -m pytest tests/entitynet/test_eval_tasks_flickr.py -sx

# # resulting file structure
# clip_benchmark/flickr30k/
#   images/*.jpg
#   flickr30k_<Split>_karpathy.txt

# ----- XM3600
# run the test, it will setup the dataset
python -m pytest tests/entitynet/test_eval_tasks_xm3600.py -sx

# # resulting file structure
# clip_benchmark/crossmodal3600/
#   xm3600/
#     captions.jsonl
#     images/*.jpg
#   crossmodal3600_captions-en.json

```

### Setup EntityNet dataset from URLs

Note that the default settings for the scripts below is to download the validation set only.

Use `--splits val,test,minitrain,train` to download all splits.

```bash
python -m entitynet.cli.urlbuild_step1_metadata
python -m entitynet.cli.urlbuild_step2_images
python -m entitynet.cli.urlbuild_step3_tars
```

## Usage

### Evaluation

```bash
# start one of the configs. you can either modify the yaml or modify the config on the fly with -o.
# by default it will run ALL tasks listed in eval_list_all.yaml
# to run only on one task or a smaller task list, use an argument like:
-o trainer.test_task_keys=imgn_1k_val
# or:
-o trainer.test_task_keys=task_list::eval_list_objcls_imgn
# use comma to separate multiple task keys.

python -m entitynet.cli.run projects/eval_clip_entitynet_zs/eval_clip_vitb32_entitynet33m.yaml --run_id e1 --test_only

# view results:
python -m entitynet.cli.view_results -s eval_clip_entitynet_zs
# use script entitynet.cli.view_results to show the metrics.
# note that in our paper we test with different prompts and report the best result over all prompts.
```

### Training

TBD setup of the training dataset

```bash
# start the experiment and give it a unique run id
# the run id will determine the subfolder and names it in the online logger
python -m entitynet.cli.run configs/projects/subdirectory/config_name.yaml --run_id myrun123
# result will be in path
cd ${ENTITYNET_OUTPUT_DIR}/experiments/subdirectory/config_name/myrun123

# Use `-o` to change the config on the fly
-o trainer.devices=1  # modify the number of gpus to 1
-o trainer.num_sanity_val_steps=0  # disable the validation sanity check before training

# useful for debugging
-o train_task.dataset.max_shards=32  # smaller train set, minimum is n_gpus * n_dataloader_workers
-o train_task.test_task_keys=imgn_1k_val,  # only test on imagenet. note the , at the end to make it a list

--run_val  # evaluate epoch 0 and exit, useful for finetuning
--test_only  # disable creating the train dataset and just run the test
```

- **Logger:** By default, will log to CSV. To use wandb, set the environment variables `WANDB_API_KEY=afe6... WAND_PROJECT=theproject` and use flag `--vislogger wandb`. (We deprecated neptune.ai because it can lead to crashes in case of internet problems, but in the code also supports it)

## Development

### Code style

- [black](https://github.com/psf/black) with line length of 100.
- Put single line docstrings in one line `"""Example."""`
- For multi line docstrings, the arguments/returns etc. definitions should be in
  [google style format] (<https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>)
  ([more details](https://google.github.io/styleguide/pyguide.html#s3.8.1-comments-in-doc-strings))
- Put type annotations into the code as type hints, and not into the docstring.

## Acknowledgements

Please refer to our paper's acknowledgement section. Additionally we would like to acknowledge:

- [https://github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) for CLIP model implementations.
- [https://github.com/LAION-AI/CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark) for the implementation of many of the evaluation datasets.
- Everyone else in the Python, Conda, PyTorch, jupyter and general open-source community whose code we used as dependencies.

## Citation

Citation will be updated after the conference proceedings are released.

```bibtex
@inproceedings{ging2025entitynet,
  author    = {Simon Ging and Sebastian Walter and Jelena Bratulić and Johannes Dienert and Hannah Bast and Thomas Brox},
  title     = {Using Knowledge Graphs to Harvest Datasets for Efficient CLIP Model Training},
  booktitle = {Pattern Recognition - 46th {DAGM} German Conference, {DAGM} {GCPR} 2025, Freiburg, Germany, September 23-26, 2025, Proceedings), 2025},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2025},
  note      = {To appear},
}
```
