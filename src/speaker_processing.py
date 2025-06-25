import os
import torch
import librosa
import numpy as np
import tempfile
import requests
from collections import defaultdict
import torch
import librosa
from pyannote.audio import Inference
from scipy.spatial.distance import cosine
import logging
import librosa
import torch, numpy as np
from speechbrain.pretrained import EncoderClassifier
# -----------------------------------------------------------------
# Load the pyannote embedding model once globally
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_MODEL = Inference("pyannote/embedding", device=DEVICE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# at top of rp_handler.py (or speaker_processing.py)
from dotenv import load_dotenv, find_dotenv
import os
# find and load your .env file
load_dotenv(find_dotenv())
HF_TOKEN = os.getenv("HF_TOKEN") 

ecapa = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device},
)

def spk_embed(wave_16k_mono: np.ndarray) -> np.ndarray:
    """Return 192-D embedding for one mono waveform @16 kHz."""
    wav = torch.tensor(wave_16k_mono).unsqueeze(0).to(device)
    return ecapa.encode_batch(wav).squeeze(0).cpu().numpy()
# -----------------------------------------------------------------
#  Select GPU when available, otherwise fall back to CPU once
# ------------------------------------------------------------------
#
# ------------------------------------------------------------------
# ------------------------------------------------------------------
#Voice Embedding Model

# ------------------------------------------------------------------
# Helper so we never forget the new 3.x input format
def to_pyannote_dict(wf, sr=16000):
    """Return mapping accepted by pyannote.audio 3.x Inference."""
    if isinstance(wf, np.ndarray):
        wf = torch.tensor(wf, dtype=torch.float32)
    if wf.ndim == 1:                      # (time,)  →  (1, time)
        wf = wf.unsqueeze(0)
    return {"waveform": wf, "sample_rate": sr}
# ------------------------------------------------------------------
def to_numpy(arr) -> np.ndarray:
    """Return a 1-D numpy embedding whatever pyannote gives back."""
    if isinstance(arr, np.ndarray):          # already good
        return arr.flatten()
    if torch.is_tensor(arr):                 # old style (should not happen)
        return arr.detach().cpu().numpy().flatten()
    # SlidingWindowFeature → .data is an np.ndarray
    from pyannote.core import SlidingWindowFeature
    if isinstance(arr, SlidingWindowFeature):
        return arr.data.flatten()
    raise TypeError(f"Unsupported embedding type: {type(arr)}")


# Set up logging (you can adjust handlers as needed)
logger = logging.getLogger("speaker_processing")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    # Only add handlers if none exist (to avoid duplicates)
    import sys
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Global cache for computed speaker embeddings.
_SPEAKER_EMBEDDING_CACHE = {}

# ---------------------------------------------------------------------
# helper  ▸  works for both Tensor and SlidingWindowFeature
# ---------------------------------------------------------------------
def _to_numpy_flat(emb):
    """
    Return a 1‑D numpy array from either:
        - torch.Tensor
        - pyannote.core.SlidingWindowFeature
        - any object with a .data attribute that is an np.ndarray
    """
    import torch, numpy as np
    from pyannote.core import SlidingWindowFeature

    if isinstance(emb, torch.Tensor):
        return emb.detach().cpu().numpy().flatten()

    if isinstance(emb, SlidingWindowFeature):
        return emb.data.flatten()

    # generic fallback: has `.data`?
    data = getattr(emb, "data", None)
    if isinstance(data, np.ndarray):
        return data.flatten()

    raise TypeError(f"Unsupported embedding type: {type(emb)}")


def load_known_speakers_from_samples(speaker_samples,  huggingface_access_token=None):
    # Use the passed token, environment variable, or fallback

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # First try with minimal logging to use cached model
        model = Inference("pyannote/embedding", use_auth_token=huggingface_access_token, device=device)
        logger.debug("Successfully loaded pyannote embedding model")
    except Exception as e:
        logger.error(f"Failed to load pyannote embedding model: {e}", exc_info=True)
        return {}
    
    """
    For each sample in speaker_samples (list of dicts with 'url' and optional 'name' and 'file_path'),
    download the file if necessary, then compute and return a dict mapping sample names to embeddings.
    If no 'name' is provided, the file name (without extension) is used.
    Uses an in-memory cache to avoid redundant computation.
    """
    global _SPEAKER_EMBEDDING_CACHE
    known_embeddings = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = Inference("pyannote/embedding", use_auth_token=huggingface_access_token, device=device)
    except Exception as e:
        logger.error(f"Failed to load pyannote embedding model: {e}", exc_info=True)
        return {}

    for sample in speaker_samples:
        # Determine sample name: use provided name; if not, extract from URL.
        name = sample.get("name")
        url = sample.get("url")
        if not name:
            if url:
                name = os.path.splitext(os.path.basename(url))[0]
                logger.debug(f"No name provided; using '{name}' from URL.")
            else:
                logger.error(f"Skipping sample with missing name and URL: {sample}")
                continue

        # Check cache first.
        if name in _SPEAKER_EMBEDDING_CACHE:
            logger.debug(f"Using cached embedding for speaker '{name}'.")
            known_embeddings[name] = _SPEAKER_EMBEDDING_CACHE[name]
            continue

        # Determine source file: if sample has a local file_path, use that; otherwise, download.
        if sample.get("file_path"):
            filepath = sample["file_path"]
            logger.debug(f"Loading speaker sample '{name}' from local file: {filepath}")
        elif url:
            try:
                logger.debug(f"Downloading speaker sample '{name}' from URL: {url}")
                response = requests.get(url)
                response.raise_for_status()
                suffix = os.path.splitext(url)[1]
                if not suffix:
                    suffix = ".wav"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(response.content)
                    tmp.flush()
                    filepath = tmp.name
                    logger.debug(f"Downloaded sample '{name}' to temporary file: {filepath}")
            except Exception as e:
                logger.error(f"Failed to download speaker sample '{name}' from {url}: {e}", exc_info=True)
                continue
        else:
            logger.error(f"Skipping sample '{name}': no file_path or URL provided.")
            try:
                waveform, _ = librosa.load(filepath, sr=16000, mono=True)
            except Exception as e:
                logger.error(f"Failed to load audio file {filepath}: {e}", exc_info=True)
                continue

        # Process the file: load audio and compute embedding.
        try:
            waveform, sr = librosa.load(filepath, sr=16000, mono=True)
            # Compute the raw embedding from pyannote
            emb = model(to_pyannote_dict(waveform, sr))
            # Convert embedding to a 1-D numpy array
            if hasattr(emb, "data"):
                emb_np = np.mean(emb.data, axis=0)
            else:
                emb_np = emb.cpu().numpy() if isinstance(emb, torch.Tensor) else np.asarray(emb)
            # L2-normalize so all vectors have unit length
            emb_np = emb_np / np.linalg.norm(emb_np)

            # cache + store
            _SPEAKER_EMBEDDING_CACHE[name] = emb_np
            known_embeddings[name] = emb_np

            logger.debug(
                f"Computed embedding for '{name}' (norm={np.linalg.norm(emb_np):.2f}).")
        except Exception as e:
            logger.error(f"Failed to process speaker sample '{name}' from file {filepath}: {e}", exc_info=True)
        
        # If we downloaded to a temporary file, you may choose to delete it:
        if not sample.get("file_path") and url:
            if 'filepath' in locals() and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    logger.debug(f"Removed temporary file for '{name}': {filepath}")
                except Exception as e:
                    logger.warning(f"Could not remove temporary file {filepath}: {e}")
    return known_embeddings


def identify_speaker(segment_embedding, known_embeddings, threshold=0.1):
    import numpy as np

    # Ensure 1-D numpy arrays
    if isinstance(segment_embedding, np.ndarray):
        segment_embedding = segment_embedding.ravel()
    else:
        logger.error("Invalid segment_embedding type, expected numpy.ndarray")
        return "Unknown", -1

    best_match, best_similarity = "Unknown", -1.0
    for speaker, known_emb in known_embeddings.items():
        if not isinstance(known_emb, np.ndarray):
            continue
        known_emb_flat = known_emb.ravel()
        # cosine expects 1-D
        score = 1 - cosine(segment_embedding, known_emb_flat)
        if score > best_similarity:
            best_similarity, best_match = score, speaker

    return (best_match, best_similarity) if best_similarity >= threshold else ("Unknown", best_similarity)
    """
    Compare a segment embedding against known speaker embeddings.
    Returns the best matching speaker and similarity score.
    If no match exceeds the threshold, returns "Unknown" and the best similarity.
    """

from collections import defaultdict
from pyannote.core import SlidingWindowFeature


import torch
import librosa
import numpy as np
from collections import defaultdict
from datetime import datetime
from pyannote.audio import Inference
from pyannote.core import SlidingWindowFeature

def process_diarized_output(
    output: dict,
    audio_filepath: str,
    known_embeddings: dict,
    huggingface_access_token: str | None = None,
    return_logs: bool = True,
    threshold: float = 0.20,
) -> tuple[dict, dict | None]:
    """
    1) Embed each diarized segment
    2) Build a centroid per diarization label
    3) Relabel any cluster whose centroid matches a known speaker
    4) Clean up all temporary fields and ensure JSON-friendly types
    """

    log_data = {
        "segments": [],
        "centroids": {},
        "relabeling_decisions": [],
        "timestamp": datetime.now().isoformat()
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = Inference("pyannote/embedding", use_auth_token=huggingface_access_token, device=device)

    segments = output.get("segments", [])
    if not segments:
        return output, None

    # 1) Embed each diarized segment
    for seg in segments:
        seg.setdefault("speaker", "Unknown")
        start, end = seg["start"], seg["end"]
        try:
            wav, _ = librosa.load(audio_filepath, sr=16000, mono=True, offset=start, duration=end - start)
        except Exception as e:
            logger.error(f"Could not load [{start:.2f}-{end:.2f}]: {e}", exc_info=True)
            continue
        if wav.size == 0:
            continue

        emb = embedder({"waveform": torch.tensor(wav)[None], "sample_rate": 16000})
        emb = _to_numpy_flat(emb)
        emb /= np.linalg.norm(emb)
        seg["__embed__"] = emb

        log_data["segments"].append({
            "start": start,
            "end": end,
            "original_speaker": seg["speaker"],
            "embedding": emb.tolist()
        })

    # 2) build cluster centroids
    # clusters: dict[str, list[np.ndarray]] = defaultdict(list)
    # for seg in segments:
    #     clusters[seg["speaker"]].append(seg["__embed__"])

    # centroids = {
    #     lbl: np.mean(mats, axis=0) / np.linalg.norm(np.mean(mats, axis=0))
    #     for lbl, mats in clusters.items() if mats
    # }
# def process_diarized_output(
    
    # 2) build cluster centroids
    clusters: dict[str, list[np.ndarray]] = defaultdict(list)
    for seg in segments:
        clusters[seg["speaker"]].append(seg["__embed__"])

    centroids = {
        lbl: np.mean(mats, axis=0) / np.linalg.norm(np.mean(mats, axis=0))
       for lbl, mats in clusters.items() if mats
    }
    # 2) build cluster centroids (only on uniform‑length embeddings)
    clusters: dict[str, list[np.ndarray]] = defaultdict(list)
    for seg in segments:
        emb = seg.get("__embed__")
        if isinstance(emb, np.ndarray) and emb.ndim == 1:
            clusters[seg["speaker"]].append(emb)

    centroids: dict[str, np.ndarray] = {}
    for lbl, mats in clusters.items():
        # ensure we have at least one embedding
        if not mats:
            continue
        # check all embeddings have the same dimension
        dims = {emb.shape[0] for emb in mats}
        if len(dims) != 1:
            logger.warning(f"Inconsistent embedding dims for '{lbl}': {dims}, skipping centroid.")
            continue
        mat_stack = np.vstack(mats)           # shape (n_segments, dim)
        mean_emb = mat_stack.mean(axis=0)     # shape (dim,)
        centroid = mean_emb / np.linalg.norm(mean_emb)
        centroids[lbl] = centroid

    # record centroids in log_data as lists
    for lbl, centroid in centroids.items():
        log_data["centroids"][lbl] = centroid.tolist()

    # 3) relabel segments based on centroids
    for lbl, centroid in centroids.items():
        name, score = identify_speaker(centroid, known_embeddings, threshold=threshold)
        decision = {
            "original_label": lbl,
            "new_label": name,
            "similarity_score": float(score),
            "threshold": threshold,
            "relabel": name != "Unknown"
        }
        log_data["relabeling_decisions"].append(decision)

        if name == "Unknown":
            continue

        for seg in segments:
            if seg["speaker"] == lbl:
                seg["speaker"] = name
                seg["similarity"] = float(score)

    # 3) cleanup temporary embeddings and ensure JSON-safe types
    for seg in segments:
        # seg.pop("__embed__", None)
        seg["start"] = float(seg["start"])
        seg["end"] = float(seg["end"])
        seg.setdefault("similarity", None)

    if return_logs:
        return output, log_data
    else:
        return output, None



##ALTERNATE SET_UP


import torch, librosa, numpy as np
from pyannote.audio import Inference
from scipy.spatial.distance import cdist



def embed_waveform(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Return a 512-dim L2-normalized embedding for a waveform."""
    feat = EMBED_MODEL({"waveform": torch.tensor(wav).unsqueeze(0), "sample_rate": sr})
    if hasattr(feat, "data"):
        arr = feat.data.mean(axis=0)
    else:
        arr = feat.squeeze(0).cpu().numpy()
    arr = arr.astype(np.float32)
    return arr / np.linalg.norm(arr)

def enroll_profiles(profiles: list[dict]) -> dict[str, np.ndarray]:
    """
    Enroll speaker profiles from provided audio samples.
    profiles: [{"name":"Alice", "file_path":"/…/alice.wav"}, …]
    returns mapping name → 512-dim vector
    """
    embeddings = {}
    for p in profiles:
        wav, sr = librosa.load(p["file_path"], sr=16000, mono=True)
        embeddings[p["name"]] = embed_waveform(wav, sr)
    return embeddings

def identify_speakers_on_segments(
    segments: list[dict],
    audio_path: str,
    enrolled: dict[str, np.ndarray],
    threshold: float = 0.5
) -> list[dict]:
    """
    Identify speakers on diarized segments using enrolled embeddings.
    segments: [{"start": 0.00, "end": 2.34, "speaker": "..."} …]
    Modifies each dict in-place, adding 'speaker_id' and 'similarity'.
    """
    names = list(enrolled.keys())
    mat = np.stack([enrolled[n] for n in names])

    for seg in segments:
        wav, sr = librosa.load(audio_path, sr=16000, mono=True,
                               offset=seg["start"],
                               duration=seg["end"] - seg["start"])
        emb = embed_waveform(wav, sr)
        sims = 1 - cdist(emb[None,:], mat, metric="cosine")[0]
        best = sims.argmax()
        if sims[best] >= threshold:
            seg["speaker_id"] = names[best]
            seg["similarity"] = float(sims[best])
        else:
            seg["speaker_id"] = "Unknown"
            seg["similarity"] = float(sims.max())
    return segments






# def process_diarized_output(
#     output: dict,
#     audio_filepath: str,
#     known_embeddings: dict,
#     huggingface_access_token: str | None = None,
#     return_logs=True,  # Add this parameter
#     threshold: float = 0.20,
#     tuple[dict, dict | None]:
#         log_data = {
#         "segments": [],
#         "centroids": {},
#         "relabeling_decisions": [],
#         "timestamp": datetime.now().isoformat()
# ) -> dict:
#     """
#     1) Embed each diarized segment
#     2) Build a centroid per diarization label
#     3) Relabel any cluster whose centroid matches a known speaker
#     4) Clean up all temporary fields and ensure JSON-friendly types
#     """

#     }

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     embedder = Inference("pyannote/embedding", use_auth_token=huggingface_access_token, device=device)

#     segments = output.get("segments", [])
#     if not segments:
#         return output

#     # 1) embed each diarized segment
#     for seg in segments:
#         # ensure every segment has a speaker key
#         seg.setdefault("speaker", "Unknown")

#         start, end = seg["start"], seg["end"]
#         try:
#             wav, _ = librosa.load(audio_filepath, sr=16000, mono=True,
#                                   offset=start, duration=end - start)
#         except Exception as e:
#             logger.error(f"Could not load [{start:.2f}-{end:.2f}]: {e}", exc_info=True)
#             continue
#         if wav.size == 0:
#             continue

#         # compute embedding (this returns a numpy array or SlidingWindowFeature)
#         emb = embedder({"waveform": torch.tensor(wav)[None], "sample_rate": 16000})
#         # convert to flat numpy
#         if hasattr(emb, "data"):
#             emb = emb.data.mean(axis=0)
#         emb = emb.cpu().numpy() if isinstance(emb, torch.Tensor) else np.asarray(emb)
#         emb = emb.flatten().astype(np.float32)
#         # L2-normalize
#         emb = emb / np.linalg.norm(emb)
#         seg["__embed__"] = emb

#     # 2) build cluster centroids
#     clusters: dict[str, list[np.ndarray]] = defaultdict(list)
#     for seg in segments:
#         clusters[seg["speaker"]].append(seg["__embed__"])

#     centroids = {
#         lbl: np.mean(mats, axis=0)
#               / np.linalg.norm(np.mean(mats, axis=0))
#         for lbl, mats in clusters.items()
#         if mats
#     }

#     # 3) relabel
#     for lbl, centroid in centroids.items():
#         log_data["centroids"][lbl] = centroid.tolist(
#         name, score = identify_speaker(centroid, known_embeddings, threshold=threshold)
#         decision = {
#             "original_label": lbl,
#             "new_label": name,
#             "similarity_score": float(score),
#             "threshold": threshold,
#             "relabel": name != "Unknown"
#         }
#         log_data["relabeling_decisions"].append(decision)
#         if name == "Unknown":
#             continue
#         # propagate label & similarity
#         for seg in segments:
#             if seg["speaker"] == lbl:
#                 seg["speaker"]    = name
#                 seg["similarity"] = float(score)

#     # 4) drop embeddings and ensure JSON-safe
#     for seg in segments:
#         seg.pop("__embed__", None)
#         # ensure start/end are floats
#         seg["start"] = float(seg["start"])
#         seg["end"  ] = float(seg["end"])
#         # similarity may be missing
#         if "similarity" not in seg:
#             seg["similarity"] = None
            
#      # Save log data to JSON file
#     # Instead of writing to file, return log data directly if requested
#     if return_logs:
#         return output, log_data
#     else:
#         return output, None


#     for segment in segments:
#         start, end = segment["start"], segment["end"]
#         duration = end - start
#         try:
#             wav16, _ = librosa.load(
#                 audio_filepath, sr=16000, mono=True, offset=start, duration=duration
#             )
#         except Exception as exc:
#             logger.error(
#                 f"Could not load {audio_filepath} [{start:.2f}–{end:.2f}s]: {exc}",
#                 exc_info=True,
#             )
#             continue
#         if wav16.size == 0:
#             continue

#         emb = spk_embed(wav16)                       # 192-d ECAPA numpy array
#         if isinstance(emb, SlidingWindowFeature):    # safety  should not happen with ECAPA
#             emb = emb.data.mean(axis=0)

#         segment["embedding"] = emb / np.linalg.norm(emb)

#     # ---------- 2) centroid per diarization label --------------------
#     cluster_embs: dict[str, np.ndarray] = defaultdict(list)
#     for seg in segments:
#         cluster_embs[seg["speaker"]].append(seg["embedding"])

#     cluster_centroids = {
#         lbl: np.mean(v, axis=0) / np.linalg.norm(np.mean(v, axis=0))
#         for lbl, v in cluster_embs.items()
#     }

#     # ---------- 3) label propagation --------------------------------
#     for lbl, centroid in cluster_centroids.items():
#         name, score = identify_speaker(centroid, known_embeddings, threshold=0.1.
        
#         decision = {
#             "original_label": lbl,
#             "new_label": name,
#             "similarity_score": float(score),
#             "threshold": threshold,
#             "relabel": name != "Unknown"
#         }
#         if name == "Unknown":
#             continue

#         # propagate to every segment (+optional word-level)
#         for seg in segments:
#             if seg["speaker"] != lbl:
#                 continue
#             seg["speaker"] = name
#             seg["similarity"] = float(best_similarity)
#             if "words" in seg:                          # word-level tags
#                 for w in seg["words"]:
#                     w["speaker"] = name

#     # embeddings are heavy  drop them before returning
#     for seg in segments:
#         seg.pop("embedding", None)

#     return output








# def process_diarized_output(output, audio_filepath, known_embeddings, huggingface_access_token=None):
#     """
#     For each diarized segment in the output, extract its audio from the given audio file,
#     compute its embedding, and update the segment's speaker label using known_embeddings.
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     embedding_model = Inference("pyannote/embedding", use_auth_token=huggingface_access_token, device=device)
    
#     segments = output.get("segments", [])
#     for segment in segments:
#         start = segment.get("start")
#         end = segment.get("end")
#         duration = end - start
#         try:
#             waveform, _ = librosa.load(audio_filepath, sr=16000, mono=True, offset=start, duration=duration)
#         except Exception as e:
#             logger.error(f"Failed to load audio segment from {audio_filepath} for {start}-{end}: {e}", exc_info=True)
#             continue
#         if len(waveform) == 0:
#             logger.warning(f"Empty waveform for segment {start}-{end} in file {audio_filepath}.")
#             continue

#         #seg_emb = embedding_model(to_pyannote_dict(waveform, 16000))
#         seg_emb = spk_embed(waveform)          # one vector per segment
#         seg["embedding"] = seg_emb        
#         from pyannote.core import SlidingWindowFeature
#         if isinstance(seg_emb, SlidingWindowFeature):
#             seg_emb = seg_emb.data.mean(axis=0)
        
#         seg_emb = to_numpy(seg_emb)
#         #seg_emb = seg_emb.detach().cpu().numpy().flatten()

# cluster_embs = defaultdict(list)
# for seg in segments:
#     cluster_embs[seg["speaker"]].append(seg["embedding"])
# cluster_embs = {k: np.mean(v, axis=0) / np.linalg.norm(np.mean(v, axis=0))
#                 for k, v in cluster_embs.items()}

# for spk_lbl, centroid in cluster_embs.items():
#     name, score = identify_speaker(centroid, known_embeddings, threshold=0.4)
#     if name != "Unknown":
#         for seg in segments:
#             if seg["speaker"] == spk_lbl:
#                 seg["speaker"] = name
#                 seg["similarity"] = float(score)

#         speaker, similarity = identify_speaker(seg_emb, known_embeddings)
#         segment["speaker"] = speaker
#         segment["similarity"] = similarity
#         logger.debug(f"Segment {start}-{end}: identified speaker '{speaker}' with similarity {similarity:.2f}.")
#     return output