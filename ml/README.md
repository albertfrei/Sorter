# Train sound traits locally

This is a first **multi-label drum one-shot baseline**, separate from the WPF app. It learns from audio, not filenames. One sample can be soft, acoustic and bright simultaneously. Training, listening and prediction stay on your machine; no cloud audio API or GPU is required.

The WPF Scan button still uses metadata/name rules. This tooling does not yet attach predictions to its table or rearrange files. First collect labels and establish useful held-out accuracy, then integrate the model. No useful model has been trained on your library yet.

## Windows quick start

Install Python 3.12 (64-bit) with the Python launcher, then double-click **`ml/Start Labeling.cmd`**. The launcher sets up/refreshes the Python environment and opens the local labeling page. Internet is needed for dependency installation; audio is never uploaded. Keep the terminal window open while labeling. Windows launcher and native folder browsing require validation on Windows; the UI flows are tested on Linux.

In the page:

1. Click **Browse folders…** (or paste the folder path).
2. Optionally enter an original pack name, then click **Import / resume folder**. Audio stays in place; import adds references to the dataset.
3. Play a sound, choose all audible traits, confirm you listened, and click **Save & next**. Labels are written immediately and the next sound appears automatically.
4. Use **Skip for now** when uncertain. Skips remain unlabeled, earn no progress, and can be revisited at the end of the round.
5. Work toward the 20-sample session goal. You can stop anytime and **Resume saved dataset** later; unfinished sounds appear first. **Trait guide & previous labels** lets you review earlier answers.

Re-importing the same folder discovers new files while preserving labels. To combine separate packs, import the first, then check **Add this folder to the current dataset** for subsequent imports. Correct original pack names before saving: categories such as `Snares` are not pack identities. Use at least five original packs for training. There is no speed score or model-suggested answer; careful, independent labels matter more than completing a round.

Labels are stored in `%LOCALAPPDATA%\Sorter\Training` on Windows. The page displays the CSV path in **Dataset details**, and at round completion, for training and backup. Set `SORTER_DATA_DIR` before launching to choose another location. On Linux the default is `ml/data/`. Session counters and skipped queues reset when the browser session ends; reviewed labels persist. Missing or moved sample paths remain in the manifest and can be skipped; do not move samples during labeling. Use one labeler per dataset.

For manual startup after creating the virtual environment:

```powershell
.\ml\.venv\Scripts\python.exe -m streamlit run ml/label.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Existing manifests still work with `-- path/to/labels.csv` after the command above. The CLI `init` command remains available for batch workflows. The folder picker browses the computer running Python, so run locally on Windows to choose your desktop files. Keep the local-only binding. WAV and FLAC are recommended initially; codec support for other formats varies. Playback and training use the first 12 seconds.

Unreviewed rows have blank tags and are excluded from training. Saving a row marks selected tags `1` and all other tags `0`: these are explicit negative labels. If unsure about the traits, skip instead of guessing. Back up the CSV. The displayed manifest path replaces `ml/data/labels.csv` in the training command below.

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
