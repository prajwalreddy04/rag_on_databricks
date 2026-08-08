# 🤖 Enterprise RAG on Databricks: End-to-End Document Q&A Pipeline

[![Databricks](https://img.shields.io/badge/Platform-Databricks-orange?style=flat&logo=databricks)](https://databricks.com)
[![Unity Catalog](https://img.shields.io/badge/Governance-Unity%20Catalog-blue)](https://docs.databricks.com/data-governance/unity-catalog/)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-00979D?logo=mlflow)](https://mlflow.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)](https://www.python.org)

An end-to-end, production-ready **Retrieval-Augmented Generation (RAG)** pipeline built natively on **Azure Databricks**. This project demonstrates how to turn unstructured documents (PDFs, text data) into an enterprise-grade Q&A system using **Mosaic AI Vector Search**, **Delta Lake**, **Unity Catalog**, **MLflow**, and **Databricks Foundation Model APIs**.

---

## 📌 Project Overview

Standard Large Language Models (LLMs) cannot access private enterprise documents and are prone to hallucinations. This repository solves that challenge by implementing a complete **RAG architecture** directly inside the Databricks Lakehouse environment.

### Why This Architecture?
- **Native Vector Search**: Continuously auto-syncs embeddings from Unity Catalog Delta Tables without external vector databases.
- **Unified Governance**: Uses Databricks **Unity Catalog** to secure raw data, text chunks, embeddings, and registered models.
- **Full Observability**: Integrated with **MLflow Tracing** to track retrieval performance, prompts, and response latency.
- **Serverless & Scalable**: Uses Databricks Foundation Model Endpoints for fast, cost-effective LLM inference.

---

## 🏗️ System Architecture

[ PDF / Enterprise Documents ]
│
▼
Notebook 01: Ingestion & Parsing (PyPDF + LangChain Splitter)
│
▼
[ Unity Catalog Delta Table ] ── (Delta Sync) ──► [ Mosaic AI Vector Search Endpoint ]
│
[ User Query ] ──────► Notebook 03: RAG Chain ─────────────────┤
(LangChain / Databricks SDK)        │ Top-K Chunks
│                            ▼
└───► [ Databricks Foundation LLM ] ──► [ Grounded Answer ]

