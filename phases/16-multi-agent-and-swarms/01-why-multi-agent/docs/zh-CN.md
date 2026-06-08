# 为什么要 Multi-Agent？

> 一个 agent 撞墙了。聪明的做法不是更大的 agent，而是更多的 agents。

**类型：** 学习
**语言：** TypeScript
**先修：** Phase 14 (Agent Engineering)
**时间：** ~60 分钟

## 学习目标

- 识别 single-agent ceiling（context overflow、mixed expertise、sequential bottleneck），并解释什么时候拆成多个 agents 是正确选择
- 比较 orchestration patterns（pipeline、parallel fan-out、supervisor、hierarchical），并为给定 task structure 选择合适模式
- 设计一个 multi-agent system，具备清晰的 role boundaries、shared state 和 communication contract
- 分析 multi-agent complexity（latency、cost、debugging difficulty）相对 single-agent simplicity 的 tradeoffs

## 要解决的问题

你在 Phase 14 构建了一个 single agent。它能工作。它可以读文件、运行命令、调用 APIs，并推理结果。然后你把它指向一个真实 codebase：200 个文件、三种语言、依赖基础设施的测试，以及在写代码前研究外部 APIs 的要求。

agent 卡住了。不是因为 LLM 笨，而是因为任务超出了一个 agent loop 能处理的范围。context window 被文件内容填满。agent 忘记了 40 次 tool calls 之前读过什么。它试图同时当 researcher、coder 和 reviewer，并且三者都做得很糟。

这就是 single-agent ceiling。每当任务需要这些东西时，你都会撞上它：

- **超过单个 window 容量的 context** - 读取 50 个文件会越过 200k tokens
- **不同阶段需要不同 expertise** - research 需要不同于 code generation 的 prompting
- **可以并行发生的工作** - 既然能同时读三个文件，为什么要顺序读取？

## 核心概念

### Single-Agent Ceiling

single agent 是一个 loop、一个 context window、一个 system prompt。想象一下：

```text
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

三件事会破裂：

1. **Context saturation** - tool results 不断堆积。到 turn 30 时，agent 已经消耗了 150k tokens 的 file contents、command outputs 和 prior reasoning。turn 5 的关键细节会丢失。

2. **Role confusion** - 一个写着 “you are a researcher, coder, reviewer, and tester” 的 system prompt，会产生一个一半 research、一半 code、并且永远没完成 review 的 agent。

3. **Sequential bottleneck** - agent 先读 file A，再读 file B，再读 file C。三次串行 LLM calls。三次串行 tool executions。没有 parallelism。

### Multi-Agent Solution

拆分工作。给每个 agent 一个 job、一个 context window，以及一个为该 job 调好的 system prompt：

```text
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

每个 agent 都有：
- 一个 focused system prompt（“You are a code reviewer. Your only job is finding bugs.”）
- 自己的 context window（不会被其他 agents 的工作污染）
- 清晰 input/output contract（接收 research notes，输出 code）

### 真实系统中的例子

**Claude Code subagents** - 当 Claude Code 用 `Task` 生成 subagent 时，它会创建一个带 scoped task 的 child agent。parent 保持 context 干净。child 做 focused work，并返回 summary。

**Devin** - 运行 planner agent、coder agent 和 browser agent。planner 将工作拆成 steps。coder 写代码。browser 研究 documentation。每个都有 separate context。

**Multi-agent coding teams (SWE-bench)** - SWE-bench 上表现最好的系统使用 researcher 读取 codebase，planner 设计 fix，coder 实现它。Single-agent systems 分数更低。

**ChatGPT Deep Research** - 并行生成多个 search agents，每个探索不同 angle，然后 synthesize results。

### Spectrum

Multi-agent 不是二元的。它是一个 spectrum：

```text
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```

**Single agent** - 一个 loop、一个 prompt。适合 simple tasks。

**Subagents** - parent 为 focused subtasks 生成 children。parent 维护 plan。children 回报结果。这就是 Claude Code 做的事。

**Pipeline** - agents 按顺序运行。Agent A 的输出成为 Agent B 的输入。适合 staged workflows：research -> code -> review -> test。

**Team** - agents 通过 shared message bus 并行运行。每个都有 role。orchestrator 负责协调。适合同时需要不同 skills 的任务。

**Swarm** - 许多相同或近似相同的 agents，带 shared state。没有固定 orchestrator。agents 从 queue 中领取工作。适合 high-throughput parallel tasks。

### 四种 Multi-Agent Patterns

#### Pattern 1: Pipeline

```text
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

每个 agent 转换数据并向前传递。易于推理。某个 stage 失败会阻塞后续全部。

#### Pattern 2: Fan-out / Fan-in

```text
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

把工作拆给 parallel agents，再 merge results。适合能拆成 independent subtasks 的任务。

#### Pattern 3: Orchestrator-Worker

```text
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

smart orchestrator 决定做什么、委派给 workers、并 synthesize results。orchestrator 本身也是一个 agent，拥有 spawning workers 的 tools。

#### Pattern 4: Peer Swarm

```text
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

没有 central orchestrator。Agents peer-to-peer 交流。decisions 从 interaction 中 emerge。更难 debug，但可扩展到许多 agents。

### 什么时候不要用 Multi-Agent

Multi-agent 会增加 complexity。agents 之间的每条 message 都是潜在 failure point。debugging 从“读一个 conversation”变成“跨五个 agents trace messages”。

**这些情况保持 single-agent：**
- 任务适合一个 context window（working data 低于约 100k tokens）
- 不需要为不同 stages 使用不同 system prompts
- Sequential execution 已经足够快
- 任务足够简单，拆分带来的 overhead 大于价值

**Complexity cost：**
- 每个 agent boundary 都是 lossy compression step：agent A 的 full context 被 summary 成给 agent B 的 message
- Coordination logic（谁做什么、何时做、按什么顺序）本身就是 bug 来源
- Latency 增加：N agents 至少意味着 N 次 serial LLM calls，如果需要来回沟通还会更多
- Cost 成倍增加：每个 agent 独立 burning tokens

Rule of thumb：如果一个任务少于 20 次 tool calls，并且适合 100k tokens，就保持 single-agent。

## 动手实现

### Step 1：过载的 Single Agent

下面是一个试图做所有事情的 single agent。它有一个巨大的 system prompt，以及一个保存 research、code 和 reviews 的 context window：

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

这种方式的问题：
- context window 会随着每个 stage 增长。到 review step 时，它同时包含 research notes、code 和 prior reasoning。
- system prompt 很 generic。它无法为每个 stage 调优。
- 没有任何事情并行运行。

### Step 2：Specialist Agents

现在拆开。每个 agent 只拿一个 job：

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

每个 specialist 都有 focused prompt。每个都获得 clean context window，其中只有它需要的 input。

### Step 3：通过 Messages 协调

用 explicit message passing 把 specialists 接起来：

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

每个 agent 只接收 addressed to it 的 messages。没有 context pollution。researcher 的 50k tokens documentation reading 永远不会进入 reviewer 的 context。

### Step 4：比较

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

multi-agent version 使用更多 total tokens（三个 agents，三次 separate LLM calls），但每个 agent 的 context 都保持干净。每个 stage 的 quality 会提升，因为 system prompt 是 specialized 的。

## 实际使用

本课产出一个可复用 prompt，用于判断什么时候应该转向 multi-agent。见 `outputs/prompt-multi-agent-decision.md`。

## 练习

1. 添加第四个 specialist：一个 “tester” agent，接收 coder 的 code 和 reviewer 的 review feedback，然后写 tests
2. 修改 pipeline，让 reviewer 能把 feedback 发回 coder，形成 revision loop（最多 2 轮）
3. 将 sequential pipeline 转换为 fan-out：并行运行 researcher 和一个 “requirements analyzer” agent，然后在传给 coder 前 merge 它们的 outputs

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| Swarm | “A hive mind of AI agents” | 一组带 shared state 且没有 fixed leader 的 peer agents。Behavior 从 local interactions 中 emerge。 |
| Orchestrator | “The boss agent” | 工具中包含 spawning 和 managing other agents 的 agent。它 planning 和 delegates，但可能不做实际工作。 |
| Coordinator | “The traffic cop” | 一个 non-agent component（通常只是 code，不是 LLM），根据规则在 agents 之间 route messages。 |
| Consensus | “The agents agree” | 多个 agents 必须在 proceeding 前达成 agreement 的 protocol。用于 conflicting outputs 需要 resolution 时。 |
| Emergent behavior | “The agents figured it out themselves” | 来自 agent interactions、但未被显式编程的 system-level patterns。可能有用，也可能有害。 |
| Fan-out / fan-in | “Map-reduce for agents” | 将 task 拆给 parallel agents（fan-out），然后组合结果（fan-in）。 |
| Message passing | “Agents talk to each other” | agents 之间的 communication mechanism：从一个 agent 发给另一个 agent 的 structured data，用来替代 shared context windows。 |

## 延伸阅读

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) - multi-agent patterns survey
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) - Microsoft 的 multi-agent conversation framework
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code) - Claude Code 如何用 Task delegate
- [CrewAI documentation](https://docs.crewai.com/) - role-based multi-agent framework
