# Train instruments and timbre locally

This is a first **instrument and timbre audio baseline**, separate from the WPF app. It learns from audio, not filenames. Each sample has one primary instrument label and multiple optional timbre traits: for example **Snare + Acoustic + Bright**. There is one labeling flow and one training command; no separate instrument mode. Training, listening and prediction stay on your machine; no cloud audio API or GPU is required.

The WPF Scan button still uses metadata/name rules. This tooling does not yet attach predictions to its table or rearrange files. First collect labels and establish useful held-out accuracy, then integrate the model. No useful model has been trained on your library yet.

## Windows quick start

Install Python 3.12 (64-bit) with the Python launcher, then double-click **`ml/Start Labeling.cmd`**. The launcher sets up/refreshes the Python environment and opens the local labeling page. Internet is needed for dependency installation; audio is never uploaded. Keep the terminal window open while labeling. Windows launcher and native folder browsing require validation on Windows; the UI flows are tested on Linux.

In the page:

1. Click **Browse folders…** (or paste the folder path).
2. Optionally enter an original pack name, then click **Import / resume folder**. Audio stays in place; import adds references to the dataset.
3. Play a sound, select **Instrument** from the first dropdown and **Timbre** from the second, then click **Save & next**. Instrument is required; timbre supports multiple traits and may be left empty. **Use previous tags** reuses both the last saved instrument and timbre selections; edit them before saving. The button resets when importing or resuming a dataset. There is no listening confirmation checkbox.
4. Use **Skip for now** when uncertain. Skips remain unlabeled, earn no progress, and can be revisited at the end of the round.
5. Work toward the 20-sample session goal. You can stop anytime and **Resume saved dataset** later; unfinished sounds appear first. **Trait guide & previous labels** lets you review earlier answers.

Re-importing the same folder discovers new files while preserving labels. To combine separate packs, import the first, then check **Add this folder to the current dataset** for subsequent imports. Correct original pack names before saving: categories such as `Snares` are not pack identities. Use at least five original packs for training. There is no speed score or model-suggested answer; careful, independent labels matter more than completing a round.

Labels are stored in `%LOCALAPPDATA%\Sorter\Training` on Windows. The page displays the CSV path in **Dataset details**, and at round completion, for training and backup. Set `SORTER_DATA_DIR` before launching to choose another location. On Linux the default is `ml/data/`. Session counters and skipped queues reset when the browser session ends; reviewed labels persist. Missing or moved sample paths remain in the manifest and can be skipped; do not move samples during labeling. Use one labeler per dataset.

For manual startup after creating the virtual environment:

```powershell
.\ml\.venv\Scripts\python.exe -m streamlit run ml/label.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Existing manifests still work with `-- path/to/labels.csv` after the command above. The CLI `init` command remains available for batch workflows. The folder picker browses the computer running Python, so run locally on Windows to choose your desktop files. Keep the local-only binding. WAV and FLAC are recommended initially; codec support for other formats varies. Playback and training use the first 12 seconds.

Existing trait-only CSVs remain readable and retain their annotations. They return to the queue for an instrument-labeling pass. Progress counts samples with instrument labels. New columns are added on the next save: `instrument` and `traits_reviewed`.

Leaving **Timbre** empty preserves any existing traits and leaves new traits unlabeled. Select **None of these traits** to explicitly record all ten traits as negative. Selecting one or more actual traits makes the other traits explicit negatives. Do not combine None with actual traits. Instrument-only samples are not used as negative examples for timbre training. Back up the CSV. The displayed manifest path replaces `ml/data/labels.csv` in the training command below.

## Labeling guide

The instrument dropdown includes kick, snare, clap, closed/open hi-hat, cymbal, tom, percussion, bass, piano/keys, guitar, strings, brass, woodwind, synth, vocal, effects, mixed and other known sounds. Start with your drum one-shots and only expand to classes you can label consistently. Use Kick for an 808 kick and Bass for sustained 808 bass notes. Use Mixed for multiple layered instruments; skip unidentified sounds instead of assigning Other as a guess. These are broad sound-source classes, not identification of individual instruments within a full mix.

Use these timbre definitions consistently; context and loudness can affect your judgments.

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
.\ml\.venv\Scripts\python.exe ml/sorter_ml.py train ml/data/labels.csv --tags soft acoustic bright --model ml/artifacts/sorter.joblib --report ml/artifacts/report.json
```

This single command trains the instrument classifier and any trainable selected timbre traits into one artifact. `--tags` selects additional timbre traits, not a separate mode. Omit it to consider all ten traits. At least 30 instrument labels and two different instrument classes are required when instrument annotations exist; aim for hundreds of varied examples per class. Legacy trait-only manifests can still train their traits, and the report explicitly says the instrument model is not trained.

The trainer splits **whole packs** into training, validation and test sets using fixed seeds shared across both tasks. Scaling is fitted only on training data; timbre thresholds are selected on validation data. Test data is used only for final metrics. At least five original packs are required. Each labeled instrument class must appear in every split; if a class is missing, add examples across more packs. Timbre traits without 30 explicit labels or positive/negative examples in all splits are listed as skipped, not silently treated as trained. With legacy trait-only data, insufficient trait coverage still stops training. Do not repeatedly change splits until a flattering result appears.

The report includes instrument precision/recall/F1 per class, macro averages and confusion matrices (row = true class, column = predicted class, ordered by `classes`), validation/test counts, trained and skipped timbre traits, and pack split membership. Review these before relying on predictions. Exact duplicate file contents are rejected to reduce leakage, but differently encoded/trimmed duplicates and related packs still need manual attention. Maintain a further untouched library for later real-world evaluation; repeatedly evaluating the same test set can bias decisions.

By default the model uses 65 features describing spectral tone, cepstra, energy, crest factor and dynamics, with a balanced multiclass logistic regression for instruments and one balanced binary logistic regression per trainable timbre trait. It reads the first **12 seconds**, averages channels to mono, resamples to 22.05 kHz and does not peak-normalize. Long loops, stereo cancellation and subjective labels can limit accuracy. Loudness/mastering differences can become misleading shortcuts.

## Optional CLAP audio embeddings

CLAP (Contrastive Language-Audio Pretraining) can replace the handcrafted features with pretrained audio embeddings. The same reviewed CSV, instrument classifier, optional timbre classifiers, pack splits and validation-selected thresholds are used. This is supervised training on your labels, not zero-shot text tagging. The labeling page and Windows launcher require no CLAP installation and continue to work unchanged; no model suggestions are added to the labeling flow.

Install the optional dependencies in the existing Python 3.12 environment:

```powershell
.\ml\.venv\Scripts\python.exe -m pip install -r ml/requirements-clap.txt
```

Download a Hugging Face **`laion/clap-htsat-unfused`** model snapshot to a local folder (for example `ml/artifacts/clap-htsat-unfused`). It must include `config.json`, `preprocessor_config.json` and the model weights (`model.safetensors` or `pytorch_model.bin`, including any shards/index). Downloading dependencies and pretrained weights requires internet and disk space; training/prediction only load local files and never upload audio or automatically fetch weights. Use trusted model files. CPU inference needs more memory and time than the baseline; no GPU is required.

```powershell
.\ml\.venv\Scripts\python.exe ml/sorter_ml.py train ml/data/labels.csv --features clap --clap-model ml/artifacts/clap-htsat-unfused --model ml/artifacts/sorter-clap.joblib --report ml/artifacts/report-clap.json
.\ml\.venv\Scripts\python.exe ml/sorter_ml.py predict ml/artifacts/sorter-clap.joblib "D:\New Samples" --output ml/artifacts/predictions-clap.json
```

CLAP reads the first **10 seconds**, mixes channels to mono, resamples to 48 kHz and uses the checkpoint's padding/preprocessing, with no peak normalization. The frozen CPU model runs in evaluation mode; normalized audio embeddings feed the existing classifiers. Fused CLAP checkpoints are not supported. Keep a separate baseline artifact to compare held-out results on the same labels and packs; pretrained embeddings do not guarantee better accuracy.

Artifacts and reports record the feature backend, its version, local model path and SHA-256 fingerprint of the checkpoint files. Prediction automatically selects the saved backend. If you move the checkpoint, pass `predict ... --clap-model NEW_DIRECTORY` with the same files. A changed checkpoint is rejected before processing audio; retrain to change weights. Old baseline artifacts without backend metadata remain usable. Keep the checkpoint directory contents stable after training.

## Suggest tags without changing files

```powershell
.\ml\.venv\Scripts\python.exe ml/sorter_ml.py predict ml/artifacts/sorter.joblib "D:\New Samples" --output ml/artifacts/predictions.json
```

JSON contains each sample's suggested `instrument`, per-class `instrument_scores`, timbre `tags` and `scores`, and `needs_review: true`. Instrument fields appear only when the artifact includes a trained instrument classifier. The top instrument is a suggestion, including for out-of-distribution sounds; this baseline does not reliably detect unknown instruments. Scores are **not calibrated confidence**. Contradictory tags can appear; a human should review them. Decode failures are recorded per file. No folders are reorganized and no audio is modified. Only load `.joblib` models you trained or trust: that format can execute code when loaded.

Keep datasets, labels and model artifacts local; the listed `ml/data/` and `ml/artifacts/` folders are Git-ignored. Other output locations are your responsibility.

## Developer checks

```powershell
.\ml\.venv\Scripts\python.exe -m unittest discover -s ml/tests -v
```

These synthetic fixtures validate actual training/prediction, combined instrument/timbre training, pack separation, optional traits, legacy manifest migration, feature extraction, manifest safety, duplicate detection and invalid input handling. They do **not** establish accuracy on real sounds or validate the browser UI visually.
