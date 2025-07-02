# 🧠 WhisperX Worker API Server (Azure ACI Edition)

This project is an adaptation of the original WhisperX worker initially designed for RunPod, now deployed as a **permanent REST microservice** on **Azure Container Instances (ACI)** with GPU support.

---

## 🎯 Purpose

Replace the "job-based" RunPod approach with an always-on API in a secure, scalable, and controlled cloud environment. The `/transcribe` endpoint processes audio files stored in Azure Blob using WhisperX (including alignment and diarization).

---

## ⚙️ Tech Stack

- Python + FastAPI  
- Docker  
- Azure Container Registry (ACR)  
- Azure Container Instances (ACI) with GPU  
- Azure Blob Storage  
- Managed Identity (for secure keyless access)  

---

## ✅ What I Did

- Containerized the WhisperX worker into a Docker image  
- Built a lightweight FastAPI server with `/transcribe` and `/status` endpoints  
- Set up a build & push pipeline to Azure Container Registry  
- Deployed the container to GPU-enabled ACI (Standard_NC6)  
- Secured blob access using **dynamic SAS tokens** via Managed Identity  
- Refactored the original client script to communicate with the ACI API instead of RunPod  

---

## 📌 Why This Project?

This project allowed me to:

- Deepen my knowledge of cloud infrastructure using Azure (ACI, ACR, Identity, Storage)  
- Learn how to secure APIs and blob access **without exposing any secrets**  
- Migrate a machine learning inference pipeline to a **cloud-native architecture** with no third-party dependency  

---

## 🔗 Related Repository

👉 Original WhisperX worker repo: [lproux/whisperx-worker](https://github.com/lproux/whisperx-worker)

---

## 🙏 Acknowledgements

Special thanks to my friend [AI Engineer @ Microsoft] for guiding the RunPod → Azure migration process.

---


