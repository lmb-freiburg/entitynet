Source is https://github.com/LAION-AI/CLIP_benchmark

For the EntityNet release we modified some of the dataset loads to fix problems and
make them easier.

Additionally we deleted some parts (CLIP models, cli, metrics) for simplicity and only kept the datasets.

Changes include:

```bash
builder.py  # many changes and fixes, debug statements etc. the original was quite messy and buggy
# some of the datasets i added some debug statements about the paths.
flickr.[y  # added a subfolder to flickr30k so the images are not directly in root.
en_classnames.json  # added some more classnames
en_classnames_nodups.json  # added a version without duplicate names
en_zeroshot_classification_templates.py  # added this explicitly instead of as json
imagenet_wnids.py  # added these to extra file so they don't clutter build.py
```
