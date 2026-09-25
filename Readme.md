---
title: Startup Challenge Matching
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Startup-Challenge Matching ML API

ML-powered matching between government challenges and startups, built
for SIH 2026.

## Endpoints

- `GET /health` — service status check
- `POST /predict` — given a `challenge_id`, returns ranked, explained
  startup recommendations

## Interactive docs

Visit `/docs` on this Space's URL to test the API directly in your
browser.