"""
Create Webdataset for iNat21
"""

from timeit import default_timer

from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from crx.datasets.inat21 import iNat21
from entitynet.datasets.inat21_webdataset import build_inat21_webdataset


def profile_speed_inat21_webdataset_vs_jpeg():
    # split = "trainnodev"
    split = "traindev"
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    workers = 4
    batch_size = 64
    dataset, dataloader = build_inat21_webdataset(
        split=split,
        transform=transform,
        language="en",
        epoch=0,
        is_train=True,
        seed=42,
        batch_size=batch_size,
        workers=workers,
    )
    n_steps, log_step = 100, 50
    t1 = default_timer()
    for i, d in enumerate(dataloader):
        if i % log_step == 0:
            print(f"{i:5d}/{n_steps}")
        if i >= n_steps:
            break
        # image_tensor = d["image"].cuda()
    delta = default_timer() - t1
    print(f"webdataset total: {delta:.3f}s per step: {delta / n_steps:.3f}s")
    del dataset, dataloader

    ds = iNat21(
        split=split,
        transform=transform,
        language="en",
        return_dict=True,
    )
    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=workers,
        # worker_init_fn=worker_init_fn_verbose,
        shuffle=True,
    )
    t1 = default_timer()
    for i, d in enumerate(dataloader):
        if i % log_step == 0:
            print(f"{i:5d}/{n_steps}")
        if i >= n_steps:
            break
        # image_tensor = d["image"].cuda()
    delta = default_timer() - t1
    print(f"jpegs total: {delta:.3f}s per step: {delta / n_steps:.3f}s")


def main():
    profile_speed_inat21_webdataset_vs_jpeg()


if __name__ == "__main__":
    main()
