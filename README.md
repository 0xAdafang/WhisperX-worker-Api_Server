# 🧠 WhisperX Worker API Server (Azure ACI Edition)

Ce projet est une adaptation du worker WhisperX initialement prévu pour RunPod, déployé désormais comme micro-service REST permanent sur **Azure Container Instances (ACI)** avec support GPU.

---

## 🎯 Objectif

Remplacer l’approche "job-based" de RunPod par une API toujours active dans un environnement cloud sécurisé, scalable et maîtrisé. L’API `/transcribe` permet de traiter des fichiers audio stockés sur Azure Blob en utilisant WhisperX (alignement, diarisation…).

---

## ⚙️ Stack utilisée

- Python + FastAPI
- Docker
- Azure Container Registry (ACR)
- Azure Container Instances (ACI) avec GPU
- Azure Blob Storage
- Managed Identity (pour sécuriser les accès sans clés)

---

## ✅ Ce que j’ai fait

- Containerisé le worker WhisperX dans une image Docker
- Créé une API REST légère avec FastAPI (`/transcribe`, `/status`)
- Mis en place un pipeline de build & push vers Azure Container Registry
- Déployé le conteneur sur ACI avec GPU (Standard_NC6)
- Sécurisé les accès aux blobs avec **SAS dynamiques** via identité managée
- Refactoré mon script client pour dialoguer avec l’API ACI au lieu de RunPod

---

## 📌 Pourquoi ?

Ce projet m’a permis de :

- Approfondir la gestion d’infrastructure cloud avec Azure (ACI, ACR, Identity, Storage)
- Apprendre à sécuriser une API et des blobs sans exposer de secrets
- Migrer une logique d’inférence ML vers une architecture **cloud-native**, sans dépendance externe

---

## 🔗 Repo lié

👉 Le repo d’origine du worker : [lproux/whisperx-worker](https://github.com/lproux/whisperx-worker)

---

## 🙏 Remerciements

Merci à mon ami [AI Engineer @ Microsoft] pour la procédure de migration RunPod → Azure.

---


