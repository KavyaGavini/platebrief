"""Export the selected food classifier for a compatible ExecuTorch host."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import statistics
import time

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime

from ensemble import AverageProbabilities
from train import make_model


def load_classifier(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = make_model(checkpoint["architecture"], len(checkpoint["classes"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint["classes"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ensemble-with", type=Path)
    args = parser.parse_args()
    model, classes = load_classifier(args.checkpoint)
    if args.ensemble_with:
        second, other_classes = load_classifier(args.ensemble_with)
        if classes != other_classes:
            raise ValueError("Ensemble members do not use the same label order")
        model = AverageProbabilities([model, second]).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "platebrief_food.pte"
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(91)
    inputs = [torch.randn(1, 3, 224, 224, generator=generator) for _ in range(5)]
    with torch.no_grad():
        edge = to_edge_transform_and_lower(torch.export.export(model, (inputs[0],)),
                                           partitioner=[XnnpackPartitioner()])
        program = edge.to_executorch()
        with destination.open("wb") as stream:
            program.write_to_file(stream)
        runtime_program = Runtime.get().load_program(str(destination))
        method = runtime_program.load_method("forward")
        maximum = 0
        for image in inputs:
            expected = model(image)
            actual = method.execute([image])[0]
            torch.testing.assert_close(actual, expected, atol=.0003, rtol=.0003)
            maximum = max(maximum, float((expected - actual).abs().max()))
        for _ in range(10):
            method.execute([inputs[0]])
        latency = []
        for _ in range(30):
            began = time.perf_counter()
            method.execute([inputs[0]])
            latency.append((time.perf_counter() - began) * 1000)
    (args.output / "food_labels.json").write_text(json.dumps(classes, indent=2))
    metadata = {
        "runtime": importlib.metadata.version("executorch"), "input_shape": [1, 3, 224, 224],
        "input_format": "sRGB image decoded and EXIF-transposed, RGB float32 CHW, resized to 224x224",
        "mean": [.485, .456, .406], "std": [.229, .224, .225],
        "classes": len(classes), "pte_bytes": destination.stat().st_size,
        "max_abs_logit_difference": maximum, "mac_median_inference_ms": statistics.median(latency),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
