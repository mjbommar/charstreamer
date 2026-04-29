# Phase 02: Native Models

## Goal

Own the first-party CPU model path for training and inference.

## Scope

- native linear regression
- native logistic regression
- shallow decision tree
- native serialization
- model reports and threshold calibration

## Tasks

- implement dense linear algebra helpers needed for training
- implement logistic training loop
- implement calibration and evaluation utilities
- implement shallow tree grow/predict path
- define native file format

## Current state

- done: first native logistic fit loop over reusable feature matrices
- done: threshold calibration and binary evaluation utilities
- done: sample ALEA and MultiLegal training/evaluation example
- done: versioned logistic JSON artifact with schema metadata and round-trip tests
- pending: shallow trees

## Acceptance criteria

- a model can be trained from extracted features without Python
- model can be serialized and loaded back
- metrics are reproducible across runs with fixed seeds

## Risks

- solver complexity growth
- numerical instability in poorly scaled features
- tree training consuming disproportionate time before logistic is hardened
