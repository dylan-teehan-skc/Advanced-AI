# Colab + Drive setup

All three accounts save to one shared Drive folder, `aai-project`, so every run, checkpoint and feature file ends up in one place.

## One-time setup

### Folder owner (Paddy, who already has `aai-project`)
1. In Google Drive, right-click `aai-project` → **Share**.
2. Add Dylan and Eddie as **Editor** (Viewers can't save checkpoints).

### Everyone else (Dylan, Eddie)
1. Open Google Drive → **Shared with me**.
2. Right-click `aai-project` → **Organise → Add shortcut** → choose **My Drive**.
3. Check in a Colab cell:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !ls "/content/drive/MyDrive/aai-project"
   ```
   You should see `features`. If you get "No such file or directory", the shortcut is missing (step 2).

Without the shortcut, Colab can't see the folder: `drive.mount` only shows My Drive.

## Folder layout

```
aai-project/
  EuroSAT_RGB.zip     the images (copy here once)
  features/           Paddy's cached ViT features
  runs/
    frozen_s0/        one folder per run: <condition>_s<seed>
    r4_s0/
    r32_s0/
      last.pt         latest checkpoint, used to resume
      best.pt         lowest validation CE, used for the single test evaluation
```

## Every training session

1. **Runtime → Change runtime type → T4 GPU.**
2. Mount Drive (as above).
3. Get the code and the pinned versions:
   ```python
   !git clone https://github.com/dylan-teehan-skc/Advanced-AI.git /content/Advanced-AI   # first time in this session
   %cd /content/Advanced-AI
   !git pull
   !pip install -q -r requirements.txt
   ```
   Always install from `requirements.txt`, so everyone runs the same torch and transformers versions.
4. Copy the images to the Colab machine (reading thousands of JPGs straight from Drive is slow):
   ```python
   !cp "/content/drive/MyDrive/aai-project/EuroSAT_RGB.zip" /content/
   !unzip -q /content/EuroSAT_RGB.zip -d /content/eurosat
   ```
   In code, pass `root="/content/eurosat/EuroSAT_RGB"` to `EuroSATDataset`.
5. Run training with the run folder on Drive, for example `out_dir="/content/drive/MyDrive/aai-project/runs/r4_s0"`.

## If Colab disconnects

Redo steps 1–4, then rerun the **same** training cell with the **same** `out_dir`. It finds `last.pt` and carries on from the next epoch. You lose at most one epoch.

Never reuse a run folder for a different condition or seed. Resume would load the wrong checkpoint.

## Who runs what (Plan, 12–14 Oct)

- Eddie's account: rank 4, seeds 0–4
- Paddy's account: frozen, seeds 0–4, plus reruns of anything that failed
- Dylan's account: rank 32, seeds 0–4
