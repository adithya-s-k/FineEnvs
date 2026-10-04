# Security

How the explorer protects accounts, credentials and data, on the public Space and when you run it yourself.
Report a vulnerability privately to the FineEnvs team through Hugging Face rather than in a public discussion.

## Credentials

* **Sessions.** Signing in (OAuth, or a pasted token) produces one cookie, encrypted with Fernet under
  `SESSION_SECRET`, HttpOnly, Secure, valid 7 days. The token inside cannot be read back out of it.
* **On the Space, sandboxes hold no credential.** A rollout's sandbox reaches its model, and the General verifier
  its judge, through `/api/llm/<capability>/v1`. The capability is random per rollout, works only while that
  rollout is live, and only for its own agent model and judge; this server adds the user's HF token or endpoint key
  on the way out. Whatever the agent runs, it cannot read your token.
* **Locally, the token is forwarded into the sandbox.** A remote HF Sandbox can't reach your machine, so it calls
  the model directly with your token, and processes in the sandbox can read it. Use a token you can revoke, or set
  `MIMO_PUBLIC_URL` to a public tunnel to this server so the proxy is used instead.
* **Nothing secret reaches a trace.** Every event is redacted (the exact token, endpoint key and capability, and
  anything shaped like `hf_…`, `sk-…` or `Bearer …`) before it is stored. The General verifier prints the first
  characters of its judge key; that is caught too.

## Your own endpoint

This server calls a bring-your-own endpoint (to test it, list its models, proxy the agent, and for Music), so:
the URL must be `https`, every address its name resolves to must be public, and the connection goes to the address
that was checked (TLS still verified against the name), which closes DNS rebinding. Replies to probes are capped
at 4 MB, probes at 30 per account per 10 minutes, and redirects are not followed.

## Requests

* **Cross-site requests.** The cookie is `SameSite=None` so the Space works inside the huggingface.co iframe, so
  every state-changing request must come from this page (checked by `Origin`, or `Sec-Fetch-Site`).
* **Headers.** A strict Content-Security-Policy on every page (scripts only from this origin and jsDelivr, no
  inline scripts, framing only by huggingface.co and hf.space), `nosniff`, a referrer policy and a permissions
  policy. Task files and rollout artifacts are served with `Content-Security-Policy: sandbox`, so a task's HTML
  can never run with this origin.
* **Input.** Model ids, URLs and keys are length- and character-limited; task files are confined to their
  workspace; database tables are looked up, never interpolated; run ids and paths are validated.

## Rollouts

* **Private rollouts** are visible only to their owner. Rollouts from before visibility existed stay private.
* **Public rollouts** are shown without who ran them: the public view has no user, no sandbox id, no endpoint URL,
  and the owner's username and endpoint host are scrubbed from every string in the trace.
* Only the owner can stop a rollout or change its visibility. My rollouts lists only your own.
* Per account at most 4 rollouts run at once, and 24 across the Space.

## Running it yourself

Without an OAuth app the server is in local mode: your machine's HF token (`HF_TOKEN` or `hf auth login`) is the
signed-in user. To keep that safe:

* It answers only to `localhost`, `127.0.0.1` and `[::1]` (a web page can't DNS-rebind its name onto your
  machine to use your token). Set `MIMO_ALLOWED_HOSTS` to serve under another name.
* The machine's token signs in only requests from this machine. Anyone else on the network has to sign in with
  their own token. Requests that arrive through a proxy are not treated as local.
* In Docker, publish to localhost only and say so explicitly:
  `docker run -p 127.0.0.1:7860:7860 -e HF_TOKEN -e MIMO_TRUST_NETWORK=1 mimo`. Without `MIMO_TRUST_NETWORK=1`,
  requests from Docker's network must sign in like anyone else.

## Hosting your own Space

Set `SESSION_SECRET` as a Space secret (sessions otherwise use the OAuth client secret), keep the runs bucket
private, and leave `hf_oauth_scopes` at `inference-api` and `jobs`.
