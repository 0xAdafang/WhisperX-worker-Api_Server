from fastapi import FastAPI, BackgroundTasks, UploadFile, File
import uuid
from worker import run_transcription
from jobs.status_store import JobStatusStore


app = FastAPI()
job_store = JobStatusStore()

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), bg: BackgroundTasks = BackgroundTasks()):
    job_id = uuid.uuid4().hex
    job_store.set_status(job_id, "QUEUED")

    contents = await file.read()
    file_path = f"/tmp/{job_id}.wav"
    with open(file_path, "wb") as f:
        f.write(contents)

    bg.add_task(run_transcription, file_path, job_id, job_store)
    return {"id": job_id, "status": "QUEUED"}

@app.get("/status/{job_id}")
def status(job_id: str):
    return {"id": job_id, "status": job_store.get_status(job_id)}
