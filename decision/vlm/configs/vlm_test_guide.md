# JRDB T1 remote SGLang test bundle (object)

The default source contains 24 development windows, 3 real images per task,
geometry-only fields and a fixed Prompt. It is NOT semantic ground truth and
does not measure detection, tracking, rendering or navigation safety.

The self-hosted remote service is preferred for online and offline VLM inference.
Only consider the repository's local Transformers entry when the remote service
is unavailable or its required GPUs are occupied by other tasks. This portable
client does not automatically switch backends or replay failed requests.

## Client setup

1. Use an existing Python environment with httpx and jsonschema. `requirements.txt`
   is for a new isolated client environment only; do not upgrade project GPU libs.
   Prefer a Chinese mirror for client dependencies:
   `python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`.
2. Edit `server_config.json`: base_url, optional api_key, exact model ID (or empty for
   discovery), allow_insecure_http=false. URL ends in /v1, not /chat/completions.
   Prefer HTTPS or an SSH tunnel. Public HTTP requires explicit opt-in and exposes
   images/tokens. Never share the filled configuration or Key.

For the self-hosted service, keep an SSH tunnel open:
`ssh -N -L 30000:127.0.0.1:30000 USER@SERVER`, then use
`http://127.0.0.1:30000/v1` with model `qwen3.8-27b-fp8`. The server must be
started separately; SSH connectivity alone does not mean the model is ready.
Packages made with `--with-deployment` also contain `VLM部署与调用.md`,
`SGLang服务部署方案.md` and the native Conda/source-build materials. Follow
those only on the intended server; packaging does not install or start anything.

## Validation and inference

Run these commands from the extracted bundle directory. Offline validation sends
zero network requests and does not read the Key:

```bash
python benchmark.py --dry-run --limit 24
```

GET model list only (zero inference):

```bash
python benchmark.py --check --out /tmp/jrdb_check
```

Three-scene smoke test, no retries:

```bash
python benchmark.py --out /tmp/jrdb_smoke
```

Only when needed, one complete serial run:

```bash
python benchmark.py --limit 24 --out /tmp/jrdb_serial
```

Simulate arrivals once every 2 seconds (0.5 Hz, NOT 2 Hz):

```bash
python benchmark.py --mode paced --rate-hz 0.5 --limit 24 --out /tmp/jrdb_paced_2s
```

## Constraints and defaults

Every output directory must be NEW. Single in-flight request; busy paced arrivals
are dropped, not queued. JSON Schema is requested, then local Schema, PID, role,
model ID, stream completeness and non-thinking checks are mandatory. No format
repair, cloud API fallback or automatic retry. Stop on the first failed task.
Environment proxies are ignored; --proxy accepts an explicitly trusted proxy.

Defaults are exported from the project's public configuration when packaged.
benchmark_defaults.json is hash-checked before use; do not hand-edit it or mix
client files from different ZIPs. CLI flags override the exported defaults.
Current defaults: non-thinking, temperature=0, max_tokens=1024, 2-second deadline.
--deadline-s changes the completion deadline independently. If only --rate-hz
is supplied, deadline becomes 1/rate for compatibility. --timeout is the HTTP
waiting limit (default 60s), NOT the result freshness deadline or robot TTL.
Offline dry-run reads public defaults/assets, never the filled server_config.
Optional --warmups 1 sends one text-only fixed-prefix warm-up, not an image replay.
The first sample and cached_tokens do NOT prove cold cache or prefix-cache state.
Server launch and hardware-specific tuning are manual.
Before a timing run, initialize the visual path with a few representative
multi-image cases and record those separately. A text-only warm-up does not
cover the first visual execution. Reuse valid results when nothing has changed;
retain one report and only the raw results needed to reproduce it.

## Results and limitations

Results: `model_check.json`, `settings.json`, `requests.jsonl`, `summary.json`. Separate
format validity from semantic accuracy; TTFT is not full-response latency. The
2-second period is a design target, not a demonstrated real-robot latency SLA.
