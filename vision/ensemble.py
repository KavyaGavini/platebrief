"""Compare an equally weighted two-model ensemble on the fixed benchmark folds."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from train import FoodPhotos, evaluate, make_model


class AverageProbabilities(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.candidates = nn.ModuleList(models)

    def forward(self, images):
        return torch.stack([model(images).softmax(dim=1) for model in self.candidates]).mean(dim=0).clamp_min(1e-12).log()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    metrics = json.loads((args.run / "metrics.json").read_text())
    if len(metrics) < 2:
        raise ValueError("Complete two candidate runs before evaluating the ensemble")
    manifest = json.loads(args.manifest.read_text())
    classes = manifest["classes"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    models = []
    for name in metrics:
        checkpoint = torch.load(args.run / f"{name}.pth", map_location="cpu", weights_only=True)
        if checkpoint["classes"] != classes:
            raise ValueError("Checkpoint label order differs from the dataset manifest")
        model = make_model(name, len(classes), pretrained=False)
        model.load_state_dict(checkpoint["state_dict"])
        models.append(model)
    ensemble = AverageProbabilities(models).to(device)
    loader = DataLoader(FoodPhotos(args.archive, manifest["folds"]["val"], False), batch_size=32, num_workers=2)
    result = evaluate(ensemble, loader, device, len(classes))
    result["best_single_macro_f1"] = max(v["validation_macro_f1"] for v in metrics.values())
    result["export_candidate"] = result["macro_f1"] >= result["best_single_macro_f1"] + .02
    test = DataLoader(FoodPhotos(args.archive, manifest["folds"]["test"], False),
                      batch_size=32, num_workers=2)
    result["test"] = evaluate(ensemble, test, device, len(classes))
    result["models"] = list(metrics)
    (args.run / "ensemble_validation.json").write_text(json.dumps(result, indent=2))
    print({k: v for k, v in result.items() if k not in {"confusion", "test"}})


if __name__ == "__main__":
    main()
