# Team Zenloop — GHCI Hackathon  
### **AI-Powered Financial Transaction Categorization**

## Note: The GHCI.iypnb file contains the entire codebase

We, **Team Zenloop**, are building a smart, hybrid **BERT + LLM** system that cleans up noisy bank transaction text and turns it into clear, meaningful categories.

# Automated AI-Based Financial Transaction Categorisation System

## Overview
This project implements an offline Automated AI-Based Financial Transaction Categorisation System focused on Indian financial transaction text such as UPI, wallet payments, and banking descriptors. These transaction strings are often noisy, unstructured, and inconsistent, making it difficult to extract meaningful category information without manual review or paid third-party APIs.

Our goal was to build a low-cost, privacy-preserving solution that can categorise transactions locally without relying on external services.

## Problem Statement
Modern financial applications require transaction categorisation for:

- budgeting and expense breakdown
- analytics and reporting
- fraud awareness
- personalised insights

However, transaction text commonly appears in formats like:

UPI//STARBUCKS BLR 45
MOBIKWIK AMAZON 450
FASTAG TOLL NH48
BPCL 300


These strings contain abbreviations, inconsistent merchant names, and lack standard structure. Existing solutions depend on third-party APIs, leading to:

- recurring costs
- dependency on external infrastructure
- privacy concerns
- limited customisation

## Our Approach
We built a hybrid pipeline combining:

- text preprocessing
- a fine-tuned DistilBERT classifier
- keyword and rule-based mapping
- merchant dictionary lookup
- optional TinyLlama reasoning for ambiguous cases

The core logic:

1. preprocess text
2. classify using DistilBERT
3. if confidence ≥ 0.75 → return result
4. otherwise apply rules and dictionary
5. if still unclear → use LLM reasoning

This allows accurate categorisation while running fully offline.

## What We Implemented in This Codebase
This repository contains:

- a Streamlit interface for:
  - single transaction input
  - CSV batch upload
- local model loading (DistilBERT, tokenizer, label encoder)
- preprocessing functions
- rule and keyword mapping
- merchant dictionary matching
- optional LLM fallback logic
- JSON-based taxonomy configuration

There is no database; all processing occurs in-memory.

## Code Structure

app.py # Streamlit UI + pipeline
preprocessing.py # text cleaning
classifier.py # DistilBERT prediction
rules.py # keyword + merchant mapping
llm_fallback.py # TinyLlama reasoning (optional)
config/ # JSON taxonomy + mappings
models/ # model artifacts


## Running the Application

pip install -r requirements.txt
streamlit run app.py


## Summary
This project demonstrates that transaction categorisation can be performed:

- locally
- privately
- cost-effectively
- with high accuracy

by combining ML models with lightweight rule-based logic.
