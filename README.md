<div align="center">
<img src='https://raw.githubusercontent.com/lmb-freiburg/entitynet/refs/heads/main/assets/logo_cv_small.png' alt='Computer Vision, University of Freiburg' title='Computer Vision, University of Freiburg'/><br/>
<a href="https://arxiv.org/abs/2505.02746">arXiv preprint</a> —
<a href="https://huggingface.co/collections/lmb-freiburg/entitynet-6810b98ea9288fef9b6c09ca">🤗 Hugging Face models</a>
</div>

# EntityNet

*Using Knowledge Graphs to harvest datasets for efficient CLIP model training*

We use knowledge graphs and web image search to build a diverse dataset of 33M images paired with 46M texts.
We show that this dataset can be used to train a generic CLIP model in a short amount of time.

Using a 10M-image subset focused on living organisms, we train domain expert models that excel
at fine-grained classification of animals, plants, and fungi.

Stay tuned for the dataset release. For now, we have released our preprint and trained CLIP models.

## Models

Our CLIP models are available on [🤗 Hugging Face](https://huggingface.co/collections/lmb-freiburg/entitynet-6810b98ea9288fef9b6c09ca)

Models are named as follows:

- Architecture 
- Trained on EntityNet-33M (all images) or trained on LivingThings-10M (only trained on images of living organisms)
- Pretrained from scratch or finetuned

### Usage

Models can be used with [open_clip](https://github.com/mlfoundations/open_clip) as follows:

```python
import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:lmb-freiburg/CLIP-ViT-B-16-EntityNet-33M")
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-16', context_length=32)

image = preprocess(Image.open("assets/rabbit.jpg")).unsqueeze(0)
text = tokenizer(["a dog", "a cat", "a rabbit"])

with torch.no_grad(), torch.autocast("cuda"):
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    logits = (model.logit_scale * image_features @ text_features.T)
    pred_class = logits.argmax(-1).item()

print(pred_class)  # prints: 2

```

## News

- 06 May 2025: **Models** on [Hugging Face](https://huggingface.co/collections/lmb-freiburg/entitynet-6810b98ea9288fef9b6c09ca)
- 05 May 2025: **Preprint** on [arXiv](https://arxiv.org/abs/2505.02746)
- 23 April 2025: **Talk** at [Stuttgart AI, Machine Learning and Computer Vision Meetup](https://voxel51.com/computer-vision-events/stuttgart-ai-ml-and-computer-vision-meetup-april-23-2025/)
- 20 March 2025: **Poster** at [2025 ELLIS Winter School on Foundation Models](https://ivi.fnwi.uva.nl/ellis/events/2025-ellis-winter-school-on-foundation-models-fomo/)

## Release roadmap

- [X] Publish preprint
- [X] Upload CLIP models on huggingface
- [ ] Add training dataset
- [ ] Add model training code
- [ ] Add evaluation code

