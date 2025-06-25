import subprocess
import os

def run_transcription(file_path: str, job_id: str, job_store):
    try:
        job_store.set_status(job_id, "RUNNING")

        output_dir = f"/tmp/output_{job_id}"
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            "whisperx",
            file_path,
            "--model", "large-v2",
            "--language", "fr",
            "--compute_type", "float16",
            "--device", "cuda",
            "--output_dir", output_dir
        ]

        subprocess.run(cmd, check=True)

        job_store.set_status(job_id, "COMPLETED")
    except Exception as e:
        job_store.set_status(job_id, f"FAILED: {str(e)}")
