# Contributing to StarVLA

StarVLA is built by the community, for the community. Here's how you can get involved:

- **Report a bug** — Open an [Issue](https://github.com/starVLA/starVLA/issues). If it needs more context, start a Discussion.
- **Propose a feature or improvement** — Please align scope via an Issue or a short sync through our [Cooperation Form](https://forms.gle/R4VvgiVveULibTCCA) before submitting a PR.
- **Need help or want to brainstorm?** — Fill out the [Cooperation Form](https://forms.gle/R4VvgiVveULibTCCA). We host office hours every Friday afternoon.
- **Before submitting a PR**, run `make check` locally to pass formatting and lint.

---

## Repository Structure

Understanding where things live makes it easier to know which file(s) to edit.

```
starVLA/
├── README.md                         ← Project overview, news, quick links
├── docs/
│   ├── CONTRIBUTING.md               ← This file
│   ├── starVLA_guideline.md          ← End-to-end quick start (install → train → eval)
│   ├── branching_strategy.md         ← Branch model, naming conventions, PR scope rules
│   ├── PR_readme.md                  ← Step-by-step PR submission guide
│   ├── faq.md                        ← Frequently asked questions
│   ├── model_zoo.md                  ← Released checkpoints and base models
│   └── WM4A.md                       ← World-Model-for-Action architecture guide
├── examples/
│   ├── LIBERO/                       ← LIBERO benchmark: data prep, training, eval
│   ├── SimplerEnv/                   ← SimplerEnv benchmark
│   ├── Robotwin/                     ← RoboTwin benchmark
│   ├── Franka/                       ← Real-robot Franka deployment
│   └── ...                           ← Other benchmarks / environments
└── starVLA/
    ├── model/framework/              ← VLA framework implementations (QwenGR00T, QwenOFT, …)
    ├── model/modules/                ← Pluggable sub-modules (VLM, world model, …)
    ├── dataloader/                   ← Dataset loaders (LeRobot format)
    ├── training/                     ← Training entry-points
    └── config/                       ← YAML configs and DeepSpeed settings
```

**Quick orientation:**

| What you want to do | Where to look |
|---|---|
| Get started from scratch | [`docs/starVLA_guideline.md`](starVLA_guideline.md) |
| Check released models | [`docs/model_zoo.md`](model_zoo.md) |
| Understand branching / PR rules | [`docs/branching_strategy.md`](branching_strategy.md) |
| Submit a PR step-by-step | [`docs/PR_readme.md`](PR_readme.md) |
| Common config questions | [`docs/faq.md`](faq.md) |
| Benchmark-specific setup | `examples/<benchmark>/README.md` |

---

## Updating the Documentation

Documentation-only changes are always welcome and do not require benchmark results or checkpoints.

### Step 1 — Branch from `starVLA_dev`

```bash
git fetch upstream
git checkout -b docs/your-change-description upstream/starVLA_dev
```

Use the `docs/` prefix so reviewers can triage quickly (see [branching_strategy.md](branching_strategy.md) for naming rules).

### Step 2 — Edit the right file

| Type of change | File to edit |
|---|---|
| Project overview, news, badges | `README.md` |
| Install / train / eval walkthrough | `docs/starVLA_guideline.md` |
| Branching or PR process | `docs/branching_strategy.md` or `docs/PR_readme.md` |
| Q&A / config tips | `docs/faq.md` |
| Model or checkpoint list | `docs/model_zoo.md` |
| Benchmark-specific instructions | `examples/<benchmark>/README.md` |

If a change affects how users interact with the project (e.g., a new CLI flag, a renamed config key), update **both** the relevant `docs/` page and any affected `examples/*/README.md` to keep them consistent.

### Step 3 — Style guidelines

- Match the tone and heading structure of the file you are editing.
- Use fenced code blocks with a language tag (` ```bash `, ` ```yaml `, ` ```python `).
- Prefer relative links between docs files (e.g., `[FAQ](faq.md)`) over absolute URLs.
- Keep line length reasonable; no hard limit, but avoid very long unwrapped paragraphs.
- Use `<details>` / `<summary>` for optional or advanced content, consistent with existing usage.

### Step 4 — Verify

```bash
# Check that internal links resolve (no build step required for Markdown)
grep -rn '](docs/' README.md docs/    # spot-check relative links
```

No `make check` is required for documentation-only PRs, but please proof-read your changes.

### Step 5 — Open a PR targeting `starVLA_dev`

Use the standard PR template. For docs-only PRs the **Testing** section can simply state:

```
- [x] Verified links and formatting manually
- No code changes — no test suite run required
```

See [`docs/PR_readme.md`](PR_readme.md) for the full PR description template and review process.
