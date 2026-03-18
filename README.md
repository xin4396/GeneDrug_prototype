# GeneDrug_prototype

GeneDrug is a prototype framework for modeling drug-induced transcriptomic perturbations from baseline cellular states.

This repository contains a limited public subset of the GeneDrug project, released as a proof-of-work sample for research, academic review, and demonstration purposes. The full internal project includes additional preprocessing pipelines, model variants, training code, benchmarking scripts, and private checkpoints that are not included here.

---

## Overview

GeneDrug studies the following problem:

> Given a baseline cellular gene expression state and a candidate drug condition, predict the resulting post-perturbation transcriptional state.

The current prototype follows a modular design:

1. **Autoencoder (AE)**  
   Compress high-dimensional gene expression into a latent representation.

2. **Perturbation predictor**  
   Predict drug-conditioned latent state transitions from baseline latent states and drug features.

3. **Decoder and evaluation modules**  
   Decode predicted latent states back to expression space and benchmark prediction or retrieval performance.

This repository is intended as a research prototype rather than a production-ready software package.

---

## Repository Structure

```text
GeneDrug_prototype/
├── cfg/                               # configuration helpers
├── dataset/                           # dataset loading utilities
├── metric/                            # evaluation metrics
├── util/                              # general utility functions
├── README.md
├── bench_mark_eval4unseen_drugs.py    # unseen-drug benchmarking
├── benchmark_ae_encoder_inference.py  # AE encoder inference on cached expression data
├── benchmark_decoder_decode.py        # decode latent predictions back to expression space
├── benchmark_mlp.py                   # autoencoder model definition / related components
├── benchmark_perturbation_mlp.py      # perturbation predictor model
├── eval_main.py                       # main evaluation entry
