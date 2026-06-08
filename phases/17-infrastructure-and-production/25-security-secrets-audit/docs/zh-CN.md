# Security — Secrets、API Key Rotation、Audit Logs、Guardrails

> 通过 centralized vaults（HashiCorp Vault、AWS Secrets Manager、Azure Key Vault）消除 secret sprawl。绝不要把 credentials 存在 config files、VCS 中的 env files、spreadsheets 里。优先使用 IAM roles，而不是 static keys；CI/CD 使用 OIDC。AI-gateway pattern 是 2026 年的解决方案：apps → gateway → model provider，由 gateway 在 runtime 从 vault 拉取 credentials。在 vault 中 rotate，所有 apps 会在几分钟内拿到新 key —— 不需要 redeploys，也不需要 Slack 里问“谁有新 key”。Rotation policy ≤90 days；每次 commit 都用 TruffleHog / GitGuardian / Gitleaks 扫描。Zero-trust：MFA、SSO、RBAC/ABAC、short-lived tokens、device posture。PII scrubbing 使用 entity recognition 在转发前 mask PHI/PII；consistent tokenization（Mesh approach）将 sensitive values 映射到 stable placeholders，让 LLM 保留 code/relationship semantics。Network egress：LLM services 放在专用 VPC/VNet subnet，只白名单 `api.openai.com`、`api.anthropic.com` 等；阻断其他所有 outbound。2026 年 incident driver：Vercel supply-chain attack，通过 compromised CI/CD credentials 外泄了数千个 customer deployments 的 env vars。

**类型:** 学习
**语言:** Python（stdlib，玩具 PII-scrubber + audit-log writer）
**先修:** Phase 17 · 19（AI Gateways），Phase 17 · 13（Observability）
**时间:** ~60 分钟

## 学习目标

- 枚举四个 secret-management anti-patterns（config files in VCS、hardcoded env、spreadsheets、static keys），并说出替代方案。
- 解释 AI-gateway-pulls-from-vault pattern 为什么是 2026 production standard。
- 实现一个带 consistent tokenization（same value → same placeholder）的 PII scrubber，让语义保留下来。
- 说出 2026 Vercel supply-chain incident，以及它对 CI/CD credential hygiene 的教训。

## 要解决的问题

实习生提交了带 API keys 的 `.env`。他们很快删除了。Keys 已经在 git history 里 —— GitGuardian scan 捕获了它，而你的 rotation process 是“Slack 团队、更新 40 个 config files、redeploy all services”。8 小时后，一半 services 已上线，一半还在等 deploy windows。

另外，用户 prompts 包含“My SSN is 123-45-6789.” Prompt 发给 OpenAI。你有 BAA，但内部 policy 要求转发前 mask PII。你没有做。

再另外，你的 EKS cluster 里的 LLM pod 可以访问任意 internet host。有人通过 DNS lookup 向 attacker-controlled domain exfil data。没有任何东西阻断它。

LLM services 的 security 必须同时处理这三个 vectors。Vault-backed credentials。PII scrubbing。Network egress filtering。Audit logs。

## 核心概念

### Centralized vault + IAM-role pull

**Vault**：HashiCorp Vault、AWS Secrets Manager、Azure Key Vault、GCP Secret Manager。单一事实源。

**IAM role**：app/gateway 通过自己的 IAM identity 认证，而不是 static key。Vault 返回 token 生命周期内可用的 secret。

**AI-gateway pattern**：gateway 在 request time 从 vault 拉取 `OPENAI_API_KEY`。在 vault 中 rotate；下一个 request 得到新 key。无需 redeploys。

### Rotation policy ≤ 90 days

所有 API keys、vault root tokens、CI/CD credentials。尽可能自动 rotate。Manual rotation 要记录并追踪。

### Secret scanning

- **TruffleHog** — 对 commits 做 regex + entropy。
- **GitGuardian** — commercial，高准确率。
- **Gitleaks** — OSS，在 CI 中运行。

每次 commit 都运行。如果检测到 new secret，就 block PR。

### Zero-trust posture

- 所有 accounts 都要求 MFA。
- 通过 SAML/OIDC 使用 SSO。
- RBAC（role-based）或 ABAC（attribute-based）用于 fine grained access。
- Short-lived tokens（小时，而不是天）。
- Device posture —— 只有启用 disk encryption 的 corp devices。

### PII / PHI scrubbing

在 prompt 离开你的 infra 前：

1. Entity recognition（spaCy NER、Presidio、commercial）。
2. Mask matched entities：`"My SSN is 123-45-6789"` → `"My SSN is [SSN_TOKEN_A3F]"`。
3. Consistent tokenization（Mesh approach）：同一 value 映射到同一 placeholder，让 LLM 保留 relationships。
4. 可选的 LLM response reverse mapping。

Static regex filters 能抓住 basic patterns；NER 能抓更多。两者都用。

### Input + output guardrails

Input：阻断已知 jailbreaks、forbidden topics；按 user 做 rate-limit。

Output：对 leaked secrets 做 regex scrub（API key patterns、refusal contexts 中的 email patterns），用 classifier 检测 policy violations。

### Network egress whitelist

LLM services 放在 dedicated subnet：
- Whitelist：`api.openai.com`、`api.anthropic.com`、vector DB endpoints、vault endpoints。
- 其他所有：drop。
- DNS 通过 allowlist-only resolver（避免 DNS-tunneling exfil）。

### Audit log

每个 LLM call 的 immutable log，包含：
- Timestamp。
- User / tenant。
- Prompt hash（出于 privacy，不存 raw prompt）。
- Model + version。
- Token counts。
- Cost。
- Response hash。
- 任何 guardrail trips。

按监管要求保留（SOC 2 1 year，HIPAA 6 years）。

### 2026 Vercel incident

Supply-chain attack：compromised CI/CD credentials 外泄了数千个 customer deployments 的 env vars。教训：CI/CD credentials 等同于 prod。存入 vault。Scope narrowly。Rotate aggressively。

### 你应该记住的数字

- Rotation policy：≤ 90 days。
- 每次 commit 扫描：TruffleHog / GitGuardian / Gitleaks。
- Vercel 2026：CI/CD creds compromised → 数千个 customer env vars leaked。
- Audit log retention：SOC 2 = 1 year，HIPAA = 6 years。

## 实际使用

`code/main.py` 实现一个带 consistent tokenization 的玩具 PII scrubber 和 append-only audit log。

## 交付成果

本课产出 `outputs/skill-llm-security-plan.md`。给定 regulatory scope 和 current state，它会规划 vault migration、scrubber、egress、audit log。

## 练习

1. 运行 `code/main.py`。发送两个引用相同 SSN 的 prompts。确认二者得到相同 placeholder。
2. 为一个调用 OpenAI + Anthropic + Weaviate 的 vLLM-on-EKS deployment 设计 network egress policy。
3. 你在 git history 中发现一个 key（2 年前）。正确响应是什么 —— rotate the key、scrub history，还是二者都做？说明理由。
4. 你的 audit log 每天增长 10 GB。设计 retention tiers（hot 30d、warm 12mo、cold 6yr）。
5. 论证 reverse-tokenization（把真实 values 替换回 LLM response）相对保持 placeholders visible 是否值得这种复杂度。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|------------|----------|
| Vault | “secrets store” | Centralized credential management service |
| IAM role | “identity-based auth” | 由 app assumed 的 role；返回 short-lived creds |
| OIDC for CI/CD | “cloud-issued tokens” | CI 中没有 static keys —— 通过 OIDC 做 identity |
| TruffleHog / GitGuardian / Gitleaks | “secret scanners” | Commit-time secret detection |
| RBAC / ABAC | “access control” | Role-based vs attribute-based |
| PII scrubbing | “data masking” | 移除或 tokenize sensitive entities |
| Consistent tokenization | “stable placeholders” | 同一 value → 同一 token |
| Mesh approach | “Mesh tokenization” | 保留语义的 tokenization pattern |
| Egress whitelist | “outbound allowlist” | 只能访问 permitted domains |
| Audit log | “immutable history” | 用于 compliance 的 append-only record |

## 延伸阅读

- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — Manage LLM API keys with secret references](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM Guardrails Best Practices](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — Secrets Management Best Practices 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) — PII detection and anonymization.
- [HashiCorp Vault docs](https://developer.hashicorp.com/vault/docs)
