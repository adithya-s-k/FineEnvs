---
title: HF RL Explorer admin
emoji: 🛠️
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Admin for the HF RL Explorer (FineEnvs members only)
hf_oauth: true
hf_oauth_expiration_minutes: 480
hf_oauth_scopes:
  - read-memberships
---

# HF RL Explorer admin

Private to the FineEnvs organization. The explorer's controls: what leads the trending row, which environments are
hidden, the collections, indexing, the rollout switches, rollouts taken off Community, and every rollout with who
ran it. It shares the explorer's bucket, not its process: changes land in the bucket's settings file and the
explorer picks them up within seconds. Every change is recorded with who made it.

Space variables: `RLX_APP=app.admin_app:app`, `RLX_OAUTH_SCOPES=openid profile read-memberships` (private FineEnvs memberships count too), `RLX_EXPLORER_URL` (the
explorer's URL). Volume: the bucket `FineEnvs/rl-explorer-data` at `/data`.
