"""Compare the bundled old classifier and the new export on shared labels."""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import zipfile

from executorch.runtime import Runtime
from PIL import Image, ImageOps
import torch
from torchvision import transforms


ALIASES = {"chips-french-fries": "french_fries", "dairy-ice-cream": "ice_cream"}


def transform(size):
    return transforms.Compose([
        transforms.Resize((size, size)), transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("export", type=Path)
    parser.add_argument("legacy_assets", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    data = json.loads(args.manifest.read_text())
    classes = data["classes"]
    legacy_classes = (args.legacy_assets / "food_labels.txt").read_text().splitlines()
    mapping = {index: legacy_classes.index(label) for index, name in enumerate(classes)
               for label in [ALIASES.get(name, name.replace("-", "_"))] if label in legacy_classes}
    prepared = {}
    for name, path, size in (("legacy", args.legacy_assets / "food_cnn.pte", 128),
                             ("platebrief", args.export / "platebrief_food.pte", 224)):
        program = Runtime.get().load_program(str(path))
        prepared[name] = (program, program.load_method("forward"), transform(size))
    results = {name: Counter() for name in prepared}
    support = Counter()
    with zipfile.ZipFile(args.archive) as archive:
        for row in data["folds"]["test"]:
            index = row["label"]
            if index not in mapping:
                continue
            label = classes[index]
            support[label] += 1
            with Image.open(io.BytesIO(archive.read(row["member"]))) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
            for name, (_, method, normalize) in prepared.items():
                expected = index if name == "platebrief" else mapping[index]
                predicted = method.execute([normalize(image).unsqueeze(0)])[0].argmax().item()
                results[name][label] += int(predicted == expected)
    report = {"images": sum(support.values()), "support": support, "correct": results,
              "legacy_output_classes": len(legacy_classes), "platebrief_output_classes": len(classes)}
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
