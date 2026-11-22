# PlateBrief

Open this folder in VS Code. PlateBrief recognizes a food photo inside the existing ExecuTorch Android demo. The app asks the user to confirm the label and enter a serving weight. Kotlin computes the estimate from the local nutrition catalog and passes the confirmed details to Llama for follow-up chat.

## Dataset and evaluation

The [MyFoodRepo Food Recognition Benchmark 2022](https://www.aicrowd.com/challenges/food-recognition-benchmark-2022) provides photographs contributed by users. The [publisher's rules](https://www.aicrowd.com/challenges/food-recognition-benchmark-2022/challenge_rules) list CC BY 4.0 for the data. PlateBrief uses the Kaggle v2.0 mirror registered in `data/sources.toml` and caches the archive locally. The pipeline reads photos directly from that archive.

The manifest retains photos where the annotations agree on one dish, groups closely related labels such as pizza toppings, and filters low-support labels. A coarse thumbnail fingerprint groups duplicates before splitting and drops groups with conflicting labels. It does not cover strongly edited variants, mixed plates, or independent phone-photo field performance. The held-out fold contains 2,474 photographs from the same source. The publisher's unlabelled test fold is excluded.

## Reproduce the food benchmark

Training used Apple MPS and Python with PyTorch 2.15.0, torchvision 0.29.2, and Pillow 13.0.0. Export used ExecuTorch 1.5.1 with PyTorch 2.14.0 and torchvision 0.29.0.

```sh
mkdir -p .cache
curl -fL 'https://www.kaggle.com/api/v1/datasets/download/awsaf49/food-recognition-2022-dataset' -o .cache/myfoodrepo.zip
python vision/prepare_myfoodrepo.py .cache/myfoodrepo.zip artifacts/myfoodrepo-dishes/folds.json --minimum 90
python vision/train.py .cache/myfoodrepo.zip artifacts/myfoodrepo-dishes/folds.json artifacts/myfoodrepo-dishes --epochs 4
python vision/ensemble.py .cache/myfoodrepo.zip artifacts/myfoodrepo-dishes/folds.json artifacts/myfoodrepo-dishes
```

Select checkpoints using validation macro-F1. The ensemble averages the two models' class probabilities. It qualifies for deployment consideration if it improves validation macro-F1 by at least two percentage points. Use `vision/export.py CHECKPOINT OUTPUT_DIRECTORY` in the export environment. Pass `--ensemble-with SECOND_CHECKPOINT` for the ensemble. Reports include the confusion matrices and macro-F1 for the fixed test fold.

## Android and local Llama

The existing app ID is `com.example.executorchllamademo`. The food model, labels and shape metadata go under `executorch-demo/app/src/main/assets/platebrief/`. Build with JDK 21 and the Android SDK using `./gradlew :app:assembleDebug` from `executorch-demo`. Updating an existing install requires a matching signing certificate.

The chat model uses Llama 3.2 1B weights with a fresh ExecuTorch 1.5.1 XNNPACK export, 8da4w linear layers, 4-bit embeddings and a 1,024-token context. Load `platebrief-llama.pte` and `platebrief-llama-tokenizer.model` under **Load local model** and select **Llama 3**. Keep both files together. The app's classifier handles photos, while Llama receives the confirmed dish and serving data as text. This release covers Android chat and food photos. Voice and WhatsApp are later stages.

To reproduce the Llama export, supply the authorized Llama 3.2 1B source checkpoint, parameters, and matching tokenizer at the paths in `models/llama-android.yaml`. In the Llama export environment, run `python -m executorch.extension.llm.export.export_llm --config models/llama-android.yaml`. Use its matching tokenizer as the model's companion file. The original Llama checkpoint is required to regenerate the export. The included export and tokenizer are sufficient to run chat on the phone.

### Release selection

EfficientNet-B0 is the Android classifier. On the 2,474-photo held-out split it scored 74.49% top-1, 96.52% top-5 and 71.19% macro-F1. The ensemble improved validation macro-F1 by 0.42 percentage points and approximately doubled the exported size. The Android catalog provides portion estimates for four of the classifier's 58 labels: apple pie, French fries, ice cream and pizza.
