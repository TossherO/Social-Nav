# JRDB T1 remote SGLang test bundle (object)

The default source contains 24 development windows, 3 real images per task,
geometry-only fields and a fixed Prompt. It is NOT semantic ground truth and
does not measure detection, tracking, rendering or navigation safety.

## Client setup

1. Use an existing Python environment with httpx and jsonschema. `requirements.txt`
   is for a new isolated client environment only; do not upgrade project GPU libs.
2. Edit `server_config.json`: base_url, optional api_key, exact model ID (or empty for
   discovery), allow_insecure_http=false. URL ends in /v1, not /chat/completions.
   Prefer HTTPS or an SSH tunnel. Public HTTP requires explicit opt-in and exposes
   images/tokens. Never share the filled configuration or Key.

## Validation and inference

Run these commands from the extracted bundle directory. Offline validation sends
zero network requests and does not read the Key:

```bash
python benchmark.py --dry-run --limit 24
```

GET model list only (zero inference):

```bash
python benchmark.py --check --out results/check
```

Three-scene smoke test, no retries:

```bash
python benchmark.py --out results/smoke
```

Only when needed, one complete serial run:

```bash
python benchmark.py --limit 24 --out results/serial
```

Simulate arrivals once every 2 seconds (0.5 Hz, NOT 2 Hz):

```bash
python benchmark.py --mode paced --rate-hz 0.5 --limit 24 --out results/paced_2s
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

## Results and limitations

Results: `model_check.json`, `settings.json`, `requests.jsonl`, `summary.json`. Separate
format validity from semantic accuracy; TTFT is not full-response latency. The
2-second period is a design target, not a demonstrated real-robot latency SLA.
