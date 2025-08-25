# Offline Replay Harness (OfflineWebshopEnv + Runner) - Development Guide

This document specifies a minimal, production-friendly replay harness that uses pre-generated WebShop demonstration logs
(`webshop_demonstrations_{start}-{end}.json`) as an offline environment. It enables fast, deterministic testing of agents
(policy only, no real WebShop calls), including LangGraph agents.

Goals
- Replace the live WebShop environment with a deterministic, file-backed mock (OfflineWebshopEnv).
- Provide a simple Runner that steps through an episode by consuming the demo logs.
- Support two testing modes:
  - Stub mode: use recorded LLM decisions; validates action formatting/mapping only.
  - LLM mode: call an LLM with recorded prompts and available actions; validates prompt/tool-calling reproducibility.
- Produce accuracy reports (step-level and episode-level), and mismatch logs for debugging.

Inputs (data contract)
- File: webshop_demonstrations_{start}-{end}.json
- Per-episode object (key fields):
  - session_id: int
  - instruction: str (free-text instruction extracted from observation)
  - trajectory: list of events
  - final_reward: float
  - success: bool
  - completed_by_backup: bool
- Step event (top-level decision step; has these fields):
  - step_number: int
  - observation_before_llm: str
  - llm_prompt: str (repr of list[dict]; parse via ast.literal_eval)
  - llm_thinking: str
  - llm_action_name: str
  - llm_action_arguments: dict
  - state: str ("Search" or "Result"; optionally "Item" if extended)
  - available_actions: list[str]
  - action_executed_in_env: str (e.g., "search[keywords...]", "click[Next >]")
  - observation_after_action: str
  - reward: float
  - done: bool
- Sub-events (non-decision detail):
  - item_page_action, detail_exploration, back_from_detail, customization_click, agent_buy, backup_agent_*.
  - The minimal harness only requires top-level decision steps; sub-events are optional to consume.

High-level architecture
- OfflineWebshopEnv: a deterministic environment backed by a single episode’s trajectory.
  - Implements reset(session_id) and step(action_str) like a gym-style env.
  - Returns (observation_after_action, reward, done, info) strictly from recorded data.
  - Enforces that the agent follows the recorded branch (cannot explore new branches offline).
- Runner: orchestrates episode playback.
  - Repeatedly: provide observation_before_llm to the policy; collect predicted action; call env.step; record metrics.
  - Two modes: Stub and LLM (see below).

Policy interface (pluggable)
- Signature (synchronous):
  predict_action(observation: str, state: str, available_actions: list[str], llm_prompt_repr: str | None) -> str
- Returns: final env action string exactly matching dataset format, e.g. "search[keywords]", "click[ItemId]", "click[Next >]", "click[Back to Search]".
- Implementations:
  - StubPolicy: use llm_action_name/arguments from the recorded step to format env action string.
  - LlmPolicy: parse llm_prompt (repr→list[dict]), send to LLM with functions mapped from available_actions, parse function-call JSON, then format.

OfflineWebshopEnv spec
- Construction
  - env = OfflineWebshopEnv(demo_path: str)
  - Loads all episodes into memory (or index for lazy load). Builds an index: session_id -> episode.
- Reset
  - reset(session_id: int) -> (observation, info)
  - Sets internal cursor to the first top-level decision step (step_number == 0).
  - Returns the observation_before_llm of that step (as observation).
  - info: {"step_index": 0, "total_steps": N}
- Step
  - step(action_str: str) -> (observation_after_action, reward, done, info)
  - Compares action_str with the recorded action_executed_in_env at current step.
    - If match: returns the recorded observation_after_action, reward, done, and advances cursor to next decision step.
    - If mismatch: behavior is configurable (default: mark mismatch, stop episode). See "Mismatch policy".
  - If there is no next decision step and not done, the env can return done=False and next observation as the next recorded observation_before_llm; but the minimal harness assumes each decision step corresponds to one env.step and advances to the next step.

Mismatch policy (configurable)
- Options:
  - "stop": terminate episode on first mismatch; record mismatch detail.
  - "allow": continue by forcing the recorded action (for analysis), but flag the step as mismatch.
- Mismatch record should include: session_id, step_number, state, expected_action, predicted_action, observation_excerpt.

Action normalization rules
- To compare predicted vs recorded action strings deterministically:
  - strip leading/trailing whitespace
  - collapse multiple spaces to a single space
  - keep exact case (or lower() both if agreed)
  - ensure angle-bracket buttons are exact: "click[< Prev]", "click[Next >]", "click[Back to Search]"

Functions mapping (tool → env action string)
- Search → search[{keywords}]
- select_item → click[{item_id}]
- Next → click[Next >]
- Prev → click[< Prev]
- Back_to_Search → click[Back to Search]
- (Item subgraph, optional): Description → click[description], Features → click[features], Reviews → click[reviews], Buy_Now → click[Buy Now]

Runner spec
- Construction
  - runner = Runner(env: OfflineWebshopEnv, policy: Policy, mismatch_policy: str = "stop")
- Run one episode
  - run_episode(session_id: int) -> EpisodeReport
    - Reset env(session_id) → obs
    - For step i in the episode’s decision steps:
      - state, available_actions, llm_prompt_repr taken from the recorded step (env provides step meta or runner reads demo directly)
      - action_pred = policy.predict_action(obs, state, available_actions, llm_prompt_repr)
      - obs_next, reward, done, info = env.step(action_pred)
      - record correctness = (normalize(action_pred) == normalize(recorded_action))
      - if mismatch and mismatch_policy == "stop": break
      - obs = obs_next; if done: break
    - Produce EpisodeReport with counts and mismatches.
- Run many episodes
  - run_all(session_ids: list[int] | None) -> SummaryReport
    - Iterates episodes, aggregates metrics.

Reports (outputs)
- Step-level: per step correctness, state, predicted vs recorded action.
- EpisodeReport:
  - session_id
  - steps_total, steps_matched, accuracy = matched/total
  - mismatches: list of {step_number, state, expected, predicted}
- SummaryReport:
  - episodes_total, episodes_run
  - total_steps, total_matched, overall_accuracy
  - accuracy_by_state: {Search: x.xx, Result: y.yy, Item: z.zz (if present)}

Minimal implementation outline (pseudocode)
```
class OfflineWebshopEnv:
    def __init__(self, demo_path):
        self.episodes = load_json(demo_path)  # list
        self.idx = {ep['session_id']: ep for ep in self.episodes}
        self.cur = None
    def reset(self, session_id):
        ep = self.idx[session_id]
        self.cur = {
            'ep': ep,
            'i': 0,
            'steps': [e for e in ep['trajectory'] if 'step_number' in e]
        }
        obs = self.cur['steps'][0]['observation_before_llm']
        return obs, {'step_index': 0, 'total_steps': len(self.cur['steps'])}
    def step(self, action_str):
        step = self.cur['steps'][self.cur['i']]
        expected = step['action_executed_in_env']
        ok = normalize(action_str) == normalize(expected)
        obs_next = step['observation_after_action']
        reward = step['reward']
        done = step['done']
        info = {'match': ok, 'step_number': step['step_number'],
                'state': step.get('state'), 'expected': expected}
        self.cur['i'] += 1
        return obs_next, reward, done, info

class StubPolicy:
    def predict_action(self, observation, state, available_actions, llm_prompt_repr, recorded_step=None):
        # Use recorded llm_action to format env action string deterministically
        name = recorded_step['llm_action_name']
        args = recorded_step.get('llm_action_arguments', {})
        return format_env_action(name, args)

class LlmPolicy:
    def __init__(self, llm_client):
        self.llm = llm_client
    def predict_action(self, observation, state, available_actions, llm_prompt_repr, recorded_step=None):
        messages = ast.literal_eval(llm_prompt_repr)
        functions = tools_for(available_actions)
        content, action = self.llm.chat(messages, functions=functions)
        return format_env_action(action['name'], action.get('arguments', {}))

class Runner:
    def __init__(self, env, policy, mismatch_policy='stop'):
        self.env = env; self.policy = policy; self.mismatch_policy = mismatch_policy
    def run_episode(self, session_id):
        obs, _ = self.env.reset(session_id)
        steps = self.env.cur['steps']
        report = {'session_id': session_id, 'steps_total': 0, 'steps_matched': 0, 'mismatches': []}
        for k, step in enumerate(steps):
            state = step.get('state'); avail = step.get('available_actions', [])
            action_pred = self.policy.predict_action(obs, state, avail, step.get('llm_prompt'), recorded_step=step)
            obs, reward, done, info = self.env.step(action_pred)
            report['steps_total'] += 1
            if info['match']:
                report['steps_matched'] += 1
            else:
                report['mismatches'].append({
                    'step_number': step['step_number'],
                    'state': state,
                    'expected': info['expected'],
                    'predicted': action_pred,
                })
                if self.mismatch_policy == 'stop':
                    break
            if done: break
        report['accuracy'] = report['steps_matched'] / max(1, report['steps_total'])
        return report
```

CLI suggestions (optional)
- Script: `python -m langgraph_agent.replay --demos webshop_demonstrations_0-3.json --mode stub`.
- Flags
  - `--sessions 0,1,2` or `--all`
  - `--mode stub|llm`
  - `--mismatch stop|allow`
  - `--report out.json`

Edge cases and notes
- If a step lacks `state`/`available_actions`, infer from observation (fallback), or skip with warning.
- Item sub-events: minimal harness ignores them; extend later to validate item-page subgraph (Description/Features/Reviews/Buy_Now).
- Backup-agent episodes: allowed; top-level steps still replay; mark `completed_by_backup` in episode summary.
- Normalization policy must be consistent with how actions were generated in demos.

How to use with LangGraph later
- Replace StubPolicy/LlmPolicy with your LangGraph policy wrapper exposing the same `predict_action` signature.
- The same Runner/Env provides comparable accuracy scores across policies (regression-friendly).

Next steps
- Implement OfflineWebshopEnv and Runner as small modules (e.g., `langgraph_agent/replay.py`).
- Start with Stub mode to achieve 100% accuracy on current demos.
- Add LLM mode and measure reproducibility; tune prompts/tools if needed.
