# SGHF Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development or superpowers:executing-plans and
> implement each task with test-driven development.

**Goal:** Build a public, standalone SGHF package that loads pyhf workspaces
and performs fast approximate fits, asymptotic inference, limits, and toys.

**Architecture:** Compile JSON once into a linear response model and reuse one
constrained quadratic engine for every inference operation. Keep pyhf out of
runtime dependencies and validate compatibility differentially in tests.

**Tech Stack:** Python 3.10+, NumPy, SciPy, pytest, Hatchling, pyhf 0.7.6
for development-only comparisons.

## Tasks

- [ ] Establish package metadata, documentation, tests, and CI.
- [ ] Compile pyhf workspaces and all standard modifiers.
- [ ] Implement bounded iterative Gaussian fits and typed results.
- [ ] Implement asymptotic test statistics, CLs, and limits.
- [ ] Implement reproducible toy generation and toy-based tests.
- [ ] Add pyhf-call guidance, examples, credits, and release metadata.
- [ ] Verify tests, wheel/sdist installation, local model regressions, and
      measured performance.
