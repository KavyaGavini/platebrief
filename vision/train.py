"""Fine-tune two classifiers on identical MyFoodRepo folds."""
import argparse
import io
import json
from pathlib import Path
import random
import zipfile

from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


ARCHITECTURES = {
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
}


def make_model(name, classes, pretrained):
    factory, weights = ARCHITECTURES[name]
    model = factory(weights=weights if pretrained else None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, classes)
    return model


class FoodPhotos(Dataset):
    def __init__(self, archive, examples, training):
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.15, .15, .12),
        ] if training else [transforms.Resize((224, 224))]
        self.transform = transforms.Compose([
            *augmentation, transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        self.path = archive
        self.archive = None
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        if self.archive is None:
            self.archive = zipfile.ZipFile(self.path)
        row = self.examples[index]
        with Image.open(io.BytesIO(self.archive.read(row["member"]))) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        return self.transform(image), row["label"]


@torch.inference_mode()
def evaluate(model, loader, device, count):
    model.eval()
    confusion = torch.zeros(count, count, dtype=torch.int64)
    top5_correct = 0
    for images, labels in loader:
        predictions = model(images.to(device)).topk(5).indices.cpu()
        bins = torch.bincount(labels * count + predictions[:, 0], minlength=count * count)
        confusion += bins.reshape(count, count)
        top5_correct += int(predictions.eq(labels[:, None]).any(1).sum())
    correct = confusion.diag()
    denominator = confusion.sum(0) + confusion.sum(1)
    total = int(confusion.sum())
    return {
        "images": total, "top1": int(correct.sum()) / total, "top5": top5_correct / total,
        "macro_f1": float((2 * correct / denominator.clamp_min(1)).mean()),
        "confusion": confusion.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--models", choices=ARCHITECTURES, nargs="+", default=list(ARCHITECTURES))
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    classes = manifest["classes"]
    count = len(classes)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_num_threads(4)

    def loader(split):
        training = split == "train"
        images = FoodPhotos(args.archive, manifest["folds"][split], training)
        return DataLoader(images, batch_size=args.batch, num_workers=args.workers, shuffle=training)

    train, val, test = (loader(split) for split in ("train", "val", "test"))
    support = torch.bincount(torch.tensor([r["label"] for r in manifest["folds"]["train"]]), minlength=count)
    weights = support.float().rsqrt()
    loss = nn.CrossEntropyLoss(weight=(weights / weights.mean()).to(device), label_smoothing=.05)
    results = {}
    print(f"Training device: {device}; classes: {count}; train images: {len(train.dataset)}", flush=True)
    for name in args.models:
        torch.manual_seed(91)
        random.seed(91)
        model = make_model(name, count, pretrained=True).to(device)
        optimizer = torch.optim.AdamW([
            {"params": model.features.parameters(), "lr": .0003},
            {"params": model.classifier.parameters(), "lr": .001},
        ])
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        checkpoint = args.output / f"{name}.pth"
        best = -1
        for epoch in range(1, args.epochs + 1):
            model.train()
            for step, (images, labels) in enumerate(train):
                optimizer.zero_grad(set_to_none=True)
                value = loss(model(images.to(device)), labels.to(device))
                value.backward()
                optimizer.step()
                if step % 100 == 0:
                    print(name, epoch, step, round(float(value.detach()), 3), flush=True)
            schedule.step()
            score = evaluate(model, val, device, count)
            print(name, epoch, "validation macro-F1", round(score["macro_f1"], 4), flush=True)
            if score["macro_f1"] > best:
                best = score["macro_f1"]
                temporary = checkpoint.with_suffix(".pth.part")
                torch.save({"architecture": name, "classes": classes, "epoch": epoch,
                            "state_dict": model.state_dict(), "validation": score}, temporary)
                temporary.replace(checkpoint)
        selected = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(selected["state_dict"])
        results[name] = {"validation_macro_f1": best, "test": evaluate(model, test, device, count)}
        (args.output / "metrics.json").write_text(json.dumps(results, indent=2))
        del model, optimizer, schedule
        if device.type == "mps":
            torch.mps.empty_cache()
    winner = max(results, key=lambda candidate: results[candidate]["validation_macro_f1"])
    (args.output / "selection.json").write_text(json.dumps({"winner": winner, "classes": classes}, indent=2))
    print("Selected by validation macro-F1:", winner, flush=True)


if __name__ == "__main__":
    main()
