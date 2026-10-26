# Train sound traits locally

This is a first **multi-label drum one-shot baseline**, separate from the WPF app. It learns from audio, not filenames. One sample can be soft, acoustic and bright simultaneously. Training, listening and prediction stay on your machine; no cloud audio API or GPU is required.

The WPF Scan button still uses metadata/name rules. This tooling does not yet attach predictions to its table or rearrange files. First collect labels and establish useful held-out accuracy, then integrate the model. No useful model has been trained on your library yet.

## Windows quick start

Install Python 3.12 (64-bit). In PowerShell at the repository root:

```powershell
py -3.12 -m venv ml/.venv
.\ml\.venv\Scripts\python.exe -m pip install -r ml/requirements.lock
.\ml\.venv\Scripts\python.exe ml/sorter_ml.py init "D:\Samples" "ml/data/labels.csv"
.\ml\.venv\Scripts\python.exe -m streamlit run ml/label.py --server.address 127.0.0.1 --browser.gatherUsageStats false -- ml/data/labels.csv
```

Streamlit opens a local listening and labeling page. Choose a sample number, listen, select **all** audible traits, check the pack name, and save. The page is for one labeler at a time. Stop it with Ctrl+C. Keep the default local-only binding. Nothing is copied or moved. Use WAV or FLAC initially for reliable browser playback; other formats depend on local decoder/browser support.

The manifest contains absolute file paths; keep the samples in place while labeling and training. Top-level folders initially supply pack names. **Correct these to original sample pack identities**: category folders such as `Snares` are not packs. Use the original unsorted library when possible. Renamed or copied versions of the same pack must have the same pack identity.

Unreviewed rows have blank tags and are excluded from training. Saving a row marks selected tags `1` and all other tags `0`: these are explicit negative labels. If unsure about the traits, leave the row unreviewed instead of guessing. Back up the CSV; re-running `init` refuses to replace an existing manifest.

## Labeling guide

Use these definitions consistently; context and loudness can affect your judgments.

| Trait | Listen for |
| --- | --- |
| soft | Gentle onset, rounded or subdued impact; not merely a low playback volume |
| punchy | Strong, distinct attack compared with the body/tail |
| bright | Prominent upper-frequency energy |
| dark | Subdued upper frequencies, low/mid emphasis |
| warm | Full low-mid body with smooth highs |
| acoustic | Audible character of a physical instrument/recording |
| electronic | Audible synthesized or electronic character |
| clean | Little audible noise, clipping or saturation |
| distorted | Audible saturation, clipping, fuzz or crunch, including intentional effects |
| noisy | Audible hiss, hum, grit or background noise |

Traits are not mutually exclusive; a layered sample can be both acoustic and electronic. These labels describe sound, not objective quality. Keep mostly drum one-shots in the initial dataset, covering kicks, snares, hats, percussion and multiple packs. Include positive and negative examples for every trait across packs. Aim for hundreds of positive examples per trait and a comparable range of negatives; the 30-row minimum is only a pipeline guard, not an accuracy guarantee.

## Train and evaluate

```powershell
.\ml\.venv\Scripts\python.exe ml/sorter_ml.py train ml/data/labels.csv --tags soft acoustic bright --model ml/artifacts/traits.joblib --report ml/artifacts/report.json
```

The example starts with soft, acoustic and bright. Omit `--tags` to train all ten traits, or select a different subset. The trainer splits **whole packs** into training, validation and test sets using fixed seeds. Scaling is fitted only on training data; thresholds are selected on validation data. Test data is used only for final metrics. At least five packs are required. Every selected trait needs both positive and negative examples in each split; if training stops, add diverse labeled packs. Do not repeatedly change splits until a flattering result appears.

The report lists split membership and per-trait precision, recall, F1, average precision, positive counts and thresholds. Review these before relying on suggestions. Exact duplicate file contents are rejected to reduce leakage, but differently encoded/trimmed duplicates and related packs still need manual attention. Maintain a further untouched library for later real-world evaluation; repeatedly evaluating the same test set can bias decisions.

The model uses 65 features describing spectral tone, cepstra, energy, crest factor and dynamics, with one balanced logistic regression per trait. It reads the first **12 seconds**, averages channels to mono, resamples to 22.05 kHz and does not peak-normalize. Long loops, stereo cancellation and subjective labels can limit accuracy. Acoustic/electronic distinctions may need a pretrained embedding model after we measure this baseline. Loudness/mastering differences can become misleading shortcuts. This is not a pretrained music-understanding model.

## Suggest tags without changing files

```powershell
.\ml\.venv\Scripts\python.exe ml/sorter_ml.py predict ml/artifacts/traits.joblib "D:\New Samples" --output ml/artifacts/predictions.json
```

JSON contains each sample's suggested tags, raw per-trait model scores and `needs_review: true`. Scores are **not calibrated confidence**. Contradictory tags can appear; a human should review them. Decode failures are recorded per file. No folders are reorganized and no audio is modified. Only load `.joblib` models you trained or trust: that format can execute code when loaded.

Keep datasets, labels and model artifacts local; the listed `ml/data/` and `ml/artifacts/` folders are Git-ignored. Other output locations are your responsibility.

## Developer checks

```powershell
.\ml\.venv\Scripts\python.exe -m unittest discover -s ml/tests -v
```

These synthetic fixtures validate actual training/prediction, pack separation, feature extraction, manifest safety, duplicate detection and invalid input handling. They do **not** establish accuracy on real sounds or validate the browser UI visually.
