# speaker_profiles.py
# speaker_profiles.py  ---------------------------------------------
import os, tempfile, requests, numpy as np, torch, librosa
from pyannote.audio import Inference

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_EMBED  = Inference(
    "pyannote/embedding",
    device=_DEVICE,
    use_auth_token=os.getenv("HF_TOKEN")
)

_CACHE = {}                                   # name → 512-D vector

# ---------------------------------------------------------------------
# 1)  Download profile audio (once)  → 128-D embedding  → cache
# ---------------------------------------------------------------------


def _l2(x: np.ndarray) -> np.ndarray:         # handy normaliser
    return x / np.linalg.norm(x)


def load_embeddings(profiles):
    """
    >>> load_embeddings([{"name":"alice","url":"https://…/alice.wav"}, …])
    returns {'alice': 512-D np.array, …}
    """
    out = {}
    for p in profiles:
        name, url = p["name"], p["url"]
        if name in _CACHE:
            out[name] = _CACHE[name]
            continue

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(requests.get(url, timeout=30).content)
            tmp.flush()
            wav, _   = librosa.load(tmp.name, sr=16_000, mono=True)
            vec      = _EMBED(torch.tensor(wav).unsqueeze(0)).cpu().numpy().flatten()
            vec      = _l2(vec)
            _CACHE[name] = vec
            out[name]   = vec
    return out



# ---------------------------------------------------------------------
# 2)  Replace diarization labels with closest profile name
# ---------------------------------------------------------------------
def relabel(diarize_df, transcription, embeds, threshold=0.75):
    """
    diarize_df   = pd.DataFrame from your DiarizationPipeline
    transcription= dict with 'segments' list   (output of WhisperX)
    embeds       = {"gin": vec128, ...}
    """
    names    = list(embeds.keys())
    vecstack = np.stack(list(embeds.values()))        # (N,128)

    for seg in transcription["segments"]:
        dia_spk = seg.get("speaker")                  # e.g. SPEAKER_00
        if not dia_spk:
            continue

        # --- approximate segment embedding: mean of word embeddings ----
        word_vecs = [w.get("embedding")
                     for w in seg.get("words", [])
                     if w.get("speaker") == dia_spk and w.get("embedding") is not None]

        if not word_vecs:
            continue

        centroid = np.mean(word_vecs, axis=0, keepdims=True)   # (1,128)
        sim      = 1 - cdist(centroid, vecstack, metric="cosine")
        best_idx = int(sim.argmax())
        if sim[0, best_idx] >= threshold:
            real = names[best_idx]
            seg["speaker"] = real
            seg["similarity"] = float(sim[0, best_idx])
            for w in seg.get("words", []):
                w["speaker"] = real
    return transcription