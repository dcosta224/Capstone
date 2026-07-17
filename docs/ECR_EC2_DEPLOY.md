# ECR push + EC2 hosting (MacroIQ MVP)

How the Capstone MVP gets from git into a runnable staging site on AWS.

Supabase remains the database. This doc covers **container build/push** and **on-demand EC2 demos** only. For S3 data/artifacts and the older git+`uv` EC2 path, see [AWS_WORKFLOW.md](AWS_WORKFLOW.md).

## Big picture

```text
  developer push to `deployment`
            │
            ▼
  GitHub Actions (.github/workflows/push-ecr.yml)
            │  docker build (linux/amd64)
            ▼
  Amazon ECR  repo: macroiq
            │  tags: :<git-sha>  and  :deployment
            │
            │  (manual — does NOT auto-start EC2)
            ▼
  EC2 staging (e.g. MacroIQDemo)
            │  docker pull + systemd (capstone-mvp-docker)
            ▼
  http://<public-ip>:8000
```

| Piece | What it does | Cost when idle |
|--------|----------------|----------------|
| **GitHub Actions → ECR** | Builds the Docker image and stores it | ECR storage only (cents) |
| **EC2** | Runs that image for demos | **Stop the instance** → ~EBS only (~$3/mo for ~40 GB) |

Pushing to `deployment` updates the image in ECR. It does **not** start EC2 or change the live site until someone runs the deploy script (or starts the instance and pulls).

## Part 1 — Automated ECR push

### Trigger

- **Branch:** `deployment` (not `main` yet)
- **Events:** `push` to `deployment`, or manual **Run workflow**
- **Workflow file:** `.github/workflows/push-ecr.yml`

### What the workflow does

1. Checks out the repo  
2. Authenticates to AWS with GitHub Secrets `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`  
3. Logs into ECR in `us-east-1`  
4. Ensures ECR repository `macroiq` exists  
5. Builds `Dockerfile` for `linux/amd64`  
6. Pushes:
   - `956314528899.dkr.ecr.us-east-1.amazonaws.com/macroiq:<12-char-sha>`
   - `…/macroiq:deployment` (moving “current staging” tag)

### GitHub setup (once)

Repo → **Settings → Secrets and variables → Actions**:

| Secret | Purpose |
|--------|---------|
| `AWS_ACCESS_KEY_ID` | IAM user that can push to ECR |
| `AWS_SECRET_ACCESS_KEY` | Matching secret |

IAM for that user needs ECR push permissions (create repo optional; push to `macroiq` required).

### Local / manual push (optional)

```bash
./scripts/deploy/push_mvp_ecr.sh
# or: ECR_REPO=macroiq AWS_REGION=us-east-1 ./scripts/deploy/push_mvp_ecr.sh
```

Same idea as CI; CI is the normal path on `deployment`.

### Image contents

- FastAPI app: `uvicorn mvp_web.server:app` on port **8000**  
- Health: `GET /health`  
- Secrets are **not** baked in — supplied at runtime via `/opt/capstone/.env` on EC2  

## Part 2 — EC2 hosting (run the ECR image)

### Preferred path (Docker from ECR)

Scripts:

| Script | Role |
|--------|------|
| `scripts/deploy/deploy_ecr_to_ec2.sh` | From your laptop: start EC2, SCP unit, pull image, restart service |
| `scripts/deploy/remote_setup_ecr.sh` | Runs **on** the instance (install Docker if needed, ECR login, pull, systemd) |
| `infra/aws/capstone-mvp-docker.service` | systemd unit: `docker pull` + `docker run` on `:8000` |
| `scripts/deploy/stop_staging.sh` | Stop EC2 (save compute) |
| `scripts/deploy/status.sh` | Instance state + health hint |

Or via the combined staging entrypoint:

```bash
./scripts/deploy/deploy_staging.sh --start-ec2 --ecr
./scripts/deploy/deploy_staging.sh --start-ec2 --ecr --stop-after
```

### Local config (`deploy/aws.env`)

Copy from example (file is **gitignored**):

```bash
cp deploy/aws.env.example deploy/aws.env
```

Important fields:

```env
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=956314528899
S3_BUCKET_RAW=capstone-956314528899-raw
S3_BUCKET_ARTIFACTS=capstone-956314528899-artifacts

EC2_ENABLED=true
EC2_INSTANCE_ID=i-...
EC2_KEY_PATH=C:/Users/you/.../capstone-staging.pem
EC2_SSH_USER=ec2-user

ECR_REPO=macroiq
ECR_IMAGE_TAG=deployment
```

Never commit `deploy/aws.env`, `.env`, or `*.pem`.

### EC2 instance requirements (checklist)

| Item | Notes |
|------|--------|
| AMI | Amazon Linux 2023 |
| Size | `t3.small` (or `t3.medium` if memory-tight) |
| Disk | **~40 GB gp3** (8 GB is too small for Docker + image) |
| Key pair | `.pem` you can SSH with; restrict file ACLs on Windows |
| IAM role | e.g. `capstone-ec2-staging` with **`AmazonEC2ContainerRegistryPullOnly`** (or project policy via `attach_ec2_ecr_iam.sh`) |
| Security group | **SSH 22** + **TCP 8000** (your IP and/or teammates’ IPs) |
| Public IP | Needed for `http://<ip>:8000`; IP may change after stop/start |

### One-time: secrets on the instance

From the repo root (PowerShell or Git Bash), replace the IP:

```bash
ssh -i /path/to/capstone-staging.pem ec2-user@<PUBLIC_IP> \
  "sudo mkdir -p /opt/capstone && sudo chown ec2-user:ec2-user /opt/capstone"

scp -i /path/to/capstone-staging.pem .env ec2-user@<PUBLIC_IP>:/opt/capstone/.env
```

`.env` must include working Supabase `PG_*` pooler settings (see `.env.example`). Prefer transaction pooler: `PG_PSQL_USE_TRANSACTION_POOLER_PORT=1`.

### Day-to-day: on / off

**On (demo / test)** — Git Bash or WSL (scripts are bash):

```bash
cd /path/to/Capstone
./scripts/deploy/deploy_ecr_to_ec2.sh --start-ec2
# confirm Y — starts instance if stopped, pulls macroiq:deployment, restarts container
```

Open: `http://<public-ip>:8000`  
(Get IP from AWS console or `./scripts/deploy/status.sh`.)

**Off (save money):**

```bash
./scripts/deploy/stop_staging.sh
# or Stop instance in the EC2 console
```

### Cost ballpark (us-east-1)

| State | Rough cost |
|--------|------------|
| EC2 **running** 24/7 (`t3.small` + public IPv4 + ~40 GB disk) | ~**$20–22/mo** |
| EC2 **stopped** (disk only) | ~**$3/mo** |
| ECR image storage (~0.6 GB) | cents/mo |
| GitHub Actions builds | GitHub minutes only |

## Custom URL (optional)

You do **not** need a paid domain to demo. Use a placeholder until the team decides whether to buy something like `macroiq.com`.

### Domain cost (if we buy one later)

| Item | Typical cost |
|------|----------------|
| `.com` registration (e.g. `macroiq.com`) | ~**$10–15 / year** (registrar-dependent; Route 53, Namecheap, Google Domains successor, etc.) |
| Other TLDs (`.app`, `.dev`, …) | Often similar or slightly higher; check before buying |
| Elastic IP (stable address for DNS) | ~**$0** while associated with a **running** instance; small hourly charge if the EIP is allocated but the instance is stopped (AWS public IPv4 / EIP rules) |
| HTTPS via **ALB + ACM** | ACM cert is free; **ALB** often ~**$16+/mo** even when EC2 is stopped — poor fit for on/off demos |
| HTTPS via **Let’s Encrypt** on EC2 | Free cert; no ALB; renewals need the instance up occasionally |

**Buying a domain alone does not host the site** — you still run EC2 (or another runtime) and point DNS at it.

### Options (pick one path)

| Option | Example URL | Cost | Pros | Cons |
|--------|-------------|------|------|------|
| **A. Raw IP** (current) | `http://3.88.199.144:8000` | $0 extra | Simplest | Ugly; IP often changes after stop/start |
| **B. Free placeholder DNS** | `http://3.88.199.144.sslip.io:8000` | $0 | No signup; shareable name | Still tied to current IP unless you use Elastic IP |
| **C. Free subdomain** (e.g. DuckDNS) | `http://macroiq-demo.duckdns.org:8000` | $0 | Stable-looking name | Extra account; update script if IP changes |
| **D. Paid domain + Elastic IP** | `http://demo.macroiq.com` or `https://…` | ~$12/yr + EC2 as today | Real brand URL; stable with EIP | Must register available name; HTTPS needs extra setup |
| **E. Paid domain + ALB + ACM** | `https://macroiq.com` | Domain + ALB monthly | “Production” HTTPS on AWS | **Ongoing ALB cost** — avoid for stop/start demos |

**Recommendation for Capstone now:** **B** (sslip.io) or **A** until the team agrees to pay for a domain. If you buy a name later, prefer **D** (Elastic IP + DNS ± Let’s Encrypt), not **E**, so stopping EC2 still keeps costs low.

### Free placeholders (no domain purchase)

| Approach | Example | Notes |
|----------|---------|--------|
| **Raw IP** | `http://3.88.199.144:8000` | Works; IP often changes after stop/start |
| **sslip.io / nip.io** | `http://3.88.199.144.sslip.io:8000` | Free DNS that resolves to that IP — no signup |
| **DuckDNS** (optional) | `http://macroiq-demo.duckdns.org:8000` | Free subdomain; update when IP changes, or pair with Elastic IP |

**sslip.io pattern:** take the public IPv4 and append `.sslip.io` (same port **8000** unless you front with nginx on 80).

### When you buy a real domain later

1. Confirm the name is **available** (e.g. `macroiq.com` may already be taken — have backups).  
2. Register it (~$10–15/year).  
3. Allocate an **Elastic IP** and associate it with the staging instance.  
4. Create a DNS **A record** → that Elastic IP.  
5. Optional HTTPS: Let’s Encrypt on the box (cheap) or ALB+ACM (costs more while left up).  

Until then, put the current demo link in `deploy/aws.env` as a reminder (gitignored):

```env
# Optional human-facing link for the team (not used by scripts yet)
STAGING_PUBLIC_URL=http://x.x.x.x.sslip.io:8000
```

Share that URL with partners; still open **TCP 8000** to their IPs in the security group.

## Legacy path (not ECR)

`load_to_ebs.sh` / `remote_setup.sh` still support **git pull + `uv` + systemd** without Docker. Prefer **`--ecr`** / `deploy_ecr_to_ec2.sh` so demos match the GitHub-built image.

## Related files

| Path | Purpose |
|------|---------|
| `Dockerfile` | MVP image definition |
| `.github/workflows/push-ecr.yml` | CI build → ECR |
| `.github/workflows/deploy-runtime.yml` | Manual checklist only (no auto live deploy) |
| `mvp_web/launch_ready.py` | `/health` → `launch` readiness block |
| `mvp_web/auth.py` | Cognito stubs; `AUTH_ENABLED=0` by default |

## Troubleshooting

| Symptom | Likely fix |
|---------|------------|
| Actions fails on AWS login | Check GitHub `AWS_*` secrets / IAM ECR push |
| Actions fails missing `.python-version` | Already fixed in Dockerfile (do not COPY missing file) |
| Deploy waits on `/health` forever | Add SG inbound **8000**; confirm `.env` on instance; `sudo systemctl status capstone-mvp-docker` |
| `password authentication failed` (Supabase) | Fix `PG_PASSWORD` / use pool user `postgres.<ref>` + port **6543** |
| SSH “UNPROTECTED PRIVATE KEY” (Windows) | Restrict `.pem` ACL to your user only (`icacls … /inheritance:r` then grant read) |
| Site OK yesterday, URL dead today | Instance stopped, or **public IP changed** after start — check console |

## Quick reference

```text
Code change → push `deployment` → Actions builds → ECR :deployment updated
                                                      ↓
                              (when you want a demo)
                         deploy_ecr_to_ec2.sh --start-ec2
                                                      ↓
                                         http://<ip>:8000
                                                      ↓
                                      stop_staging.sh when done
```
