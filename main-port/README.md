# main-port: DeepSeek-V4-Flash Ampere (SM8x) backend for vLLM main

These files add an `ampere/` hardware backend to `vllm/models/deepseek_v4/` on
current vLLM **main** (not the c2fb0133 patch overlay in this repo's root).
They let DeepSeek-V4-Flash sparse MLA run on SM8x (A100/RTX 3080), with DCP as
the framework provides it. Mirror of worktree branch `ampere-dsv4-dcp`
(vllm-project/vllm). Apply by copying over a main checkout. See
AMPERE-PORT-PLAN.md for status. WIP: pre-build.
