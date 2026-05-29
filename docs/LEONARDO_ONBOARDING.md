# Leonardo HPC Onboarding — Action Checklist

Source: AI:AT HPC Onboarding Chapter 4 — https://ai-at.eu/en/hpc-onboarding/chapter-4/
(See also Chapter 3 prerequisites: institutional registration + account
confirmation should already be done.)

> ⚠️ **Never commit any credential, token, SSH key, certificate, or invite
> link from this process.** Treat everything below as secrets. Per
> `docs/implementation-plan-en.md` ("what NOT to do") and global security
> rules.

## Per-person steps (run on your local machine)

### 1. Install the `step` client

```bash
# macOS — requires Homebrew (https://brew.sh)
brew install step
```

Other OSes: see https://smallstep.com/docs/step-ca/installation.

### 2. Bootstrap the CINECA certificate authority

```bash
step ca bootstrap \
  --ca-url=https://sshproxy.hpc.cineca.it \
  --fingerprint 2ae1543202304d3f434bdc1a2c92eff2cd2b02110206ef06317e70c1c1735ecd
```

### 3. Start the SSH agent in your shell

```bash
eval $(ssh-agent)
```

### 4. Request your short-lived SSH certificate

Replace `USER@EMAIL` with the email tied to your AI:AT / institutional
identity:

```bash
step ssh login 'USER@EMAIL' --provisioner cineca-hpc
```

You will be redirected to your institution's identity provider for
authentication. **MFA (one-time code) is mandatory.**

This drops:
- private key at `~/.ssh/LEONARDO_key`
- certificate at `~/.ssh/LEONARDO_key-cert.pub`
- both auto-renew if you keep the SSH agent running.

### 5. Configure `~/.ssh/config` (recommended)

Add the AI:AT-provided block to `~/.ssh/config` so you can `ssh LEONARDO`
instead of typing the full host every time. Keep this file local; do not
commit.

### 6. First login

```bash
ssh yourusername@login.LEONARDO.cineca.it
# or, after step 5:
ssh LEONARDO
```

If you see `REMOTE HOST IDENTIFICATION HAS CHANGED`, prune the old entry
from `~/.ssh/known_hosts` or connect to `login01-ext.LEONARDO.cineca.it`
directly.

## What this does NOT include

- The actual cluster training script — see `scripts/train_transformer.py`
  (placeholder for now).
- Slurm submission templates — pull from `docs/implementation-plan-en.md`
  §"Using Leonardo: only in phase 2" once we have a transformer to train.
- Project / account / QOS assignment — the AI:AT mentor or the CINECA team
  provides this.

## When to actually go to Leonardo

Per `docs/implementation-plan-en.md` and `docs/PIPELINE.md`: only after the
local pipeline (n-gram baseline + eval + dashboard) is green and we have a
small transformer that trains on CPU/MPS on a tiny sample. Do not debug
CUDA on the cluster before the local loop works.

## References

- AI:AT HPC Onboarding: https://ai-at.eu/en/hpc-onboarding/
- CINECA Leonardo docs: https://docs.hpc.cineca.it/hpc/leonardo.html
- `step` CLI installation: https://smallstep.com/docs/step-ca/installation
