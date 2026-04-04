# Implementation Guidelines

## Purpose

This document defines implementation process rules, testing expectations, and cost-control constraints. It is intentionally separate from architecture and project-goal documents.

## 1. Completion Rule

- A feature is complete only if all relevant tests pass.
- New features should add or update tests so later changes can detect regressions.
- Manual success alone is not enough.

## 2. Testing Policy

Tests should be added alongside each feature area:

- config loading tests
- article pulling tests
- summarization pipeline tests
- dataset generation tests
- model training smoke tests

A feature is not complete if:

- it only works on one manual run
- it has no regression coverage
- it breaks existing tests

## 3. Cheap-First Validation

Any expensive stage should first run on a deliberately small sample.

Examples:

- pull a small date range before full ingestion
- summarize a small article subset before full summarization
- run a small training smoke test before full historical training

This is mandatory when a stage depends on:

- paid APIs
- rate-limited APIs
- local LLM compute
- long-running preprocessing

## 4. Summarization Validation

Before summarizing the full corpus:

- test prompt behavior on a representative small sample
- verify output-format consistency
- verify summary persistence and resume behavior
- verify that downstream FinBERT embedding generation works on produced summaries

The project should not spend full compute on all articles until these checks pass.

## 5. Ingestion Safety Rules

Article pulling must be resumable.

Required behavior:

- write intermediate results during long pulls
- preserve already-fetched batches if a run stops midway
- avoid unnecessary refetching
- enforce configured start and end dates

## 6. Configuration Rules

Behavior should be config-driven instead of hardcoded.

At minimum:

- dataset generation settings come from YAML
- summarization settings come from YAML
- training settings come from YAML

Default configs should be safe enough for smoke tests and adjustable for full runs.

## 7. Rollout Order

Recommended implementation order:

1. config loading
2. article pull script with resume support
3. raw dataset validation tests
4. summarization pipeline on a small sample
5. FinBERT embedding generation
6. daily aggregation and feature joins
7. baseline models
8. full training pipeline

## 8. Gate For Advancing To Next Stage

A stage is ready to unlock the next one only if:

- outputs are persisted
- tests pass
- the stage is reproducible from config
- failure and retry behavior is understood