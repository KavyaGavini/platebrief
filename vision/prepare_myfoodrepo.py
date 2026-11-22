"""Index single-class MyFoodRepo photos and create deterministic, disjoint folds."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import zipfile

from PIL import Image, ImageOps


def canonical_category(label):
    if label.startswith("pizza-") and label.endswith("-baked"):
        return "pizza"
    if label in {"rice-basmati", "rice-jasmin", "rice-whole-grain", "rice-wild"}:
        return "rice"
    if label in {
        "pasta-spaghetti", "pasta-penne", "pasta-twist", "pasta-wholemeal",
        "pasta-linguini-parpadelle-tagliatelle", "pasta-noodles", "pasta-hornli",
        "pasta-in-butterfly-form-farfalle", "pasta-in-conch-form",
    }:
        return "pasta"
    return label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--minimum", type=int, default=120)
    args = parser.parse_args()
    candidates = []
    with zipfile.ZipFile(args.archive) as archive:
        names = set(archive.namelist())
        by_basename = defaultdict(list)
        for member in sorted(names):
            if PurePosixPath(member).suffix.lower() in {".jpg", ".jpeg", ".png"}:
                by_basename[PurePosixPath(member).name].append(member)
        for manifest in sorted(names):
            basename = PurePosixPath(manifest).name.lower()
            if not basename.endswith(".json") or not any(t in basename for t in ("annot", "train", "val")):
                continue
            document = json.loads(archive.read(manifest))
            if not isinstance(document, dict) or not all(k in document for k in ("images", "annotations", "categories")):
                continue
            cats = {item["id"]: canonical_category(item["name"]) for item in document["categories"]}
            labels = defaultdict(set)
            for item in document["annotations"]:
                if not item.get("iscrowd") and item["category_id"] in cats:
                    labels[item["image_id"]].add(cats[item["category_id"]])
            for item in document["images"]:
                category = labels[item["id"]]
                if len(category) != 1:
                    continue
                matches = by_basename[PurePosixPath(item["file_name"]).name]
                if len(matches) != 1:
                    continue
                candidates.append((matches[0], next(iter(category))))

        by_image = defaultdict(set)
        for member, label in candidates:
            by_image[member].add(label)
        nominal = Counter(next(iter(labels)) for labels in by_image.values() if len(labels) == 1)
        groups = defaultdict(list)
        unreadable = 0
        for member, labels in sorted(by_image.items()):
            if len(labels) != 1 or nominal[next(iter(labels))] < args.minimum:
                continue
            try:
                with Image.open(io.BytesIO(archive.read(member))) as raw:
                    image = ImageOps.exif_transpose(raw).convert("RGB")
                    thumbnail = image.resize((12, 12), Image.Resampling.BILINEAR)
                    digest = hashlib.sha256(bytes(value // 24 for value in thumbnail.tobytes())).hexdigest()
            except (OSError, ValueError):
                unreadable += 1
                continue
            groups[digest].append((member, next(iter(labels))))

    by_label = defaultdict(list)
    conflicts = 0
    for images in groups.values():
        labels = {label for _, label in images}
        if len(labels) != 1:
            conflicts += 1
            continue
        member, label = images[0]
        by_label[label].append(member)
    rng = random.Random(91)
    classes = sorted(label for label, members in by_label.items() if len(members) >= args.minimum)
    if len(classes) < 5:
        raise ValueError(f"Only {len(classes)} eligible classes from {len(candidates)} records. Inspect archive annotations.")
    folds = {"train": [], "val": [], "test": []}
    for index, label in enumerate(classes):
        members = sorted(by_label[label])
        rng.shuffle(members)
        held_out = round(.15 * len(members))
        for split, selection in (("test", members[:held_out]), ("val", members[held_out:2 * held_out]),
                                 ("train", members[2 * held_out:])):
            folds[split].extend({"member": member, "label": index} for member in selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"classes": classes, "folds": folds, "candidates": len(candidates),
              "unreadable": unreadable, "conflicting_thumbnail_groups": conflicts}
    args.output.write_text(json.dumps(report, indent=2))
    print(f"{len(classes)} classes, " + ", ".join(f"{name}={len(rows)}" for name, rows in folds.items()), flush=True)


if __name__ == "__main__":
    main()
