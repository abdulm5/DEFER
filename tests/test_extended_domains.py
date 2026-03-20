from __future__ import annotations

from defer.data.seeds import DOMAIN_TEMPLATES, generate_seed_tasks
from defer.domains.contracts import default_contracts
from defer.domains.extended_tools import EXTENDED_TOOLS
from defer.domains.state import WorldState
from defer.domains.tools import TOOLS
from defer.sim.environment import EnvironmentConfig, SimulationEnvironment
from defer.sim.scenario import Scenario
from defer.baselines.policies import DeferFullPolicy


def test_each_new_tool_executes():
    state = WorldState()
    args_map = {
        "register_webhook": {"url": "https://example.com/hook", "event_type": "push"},
        "upload_file": {"filename": "test.csv", "size_bytes": 1024, "shared": False},
        "modify_access": {"principal": "user@example.com", "resource": "db", "permission": "read"},
        "send_notification": {"message": "Hello", "channel": "email", "recipients": ["a@b.com"], "deliver_now": False},
    }
    for tool_name, tool_fn in EXTENDED_TOOLS.items():
        result = tool_fn(state, args_map[tool_name])
        assert result.ok
        assert isinstance(result.observation, dict)


def test_each_tool_has_contract():
    contracts = default_contracts()
    for tool_name in EXTENDED_TOOLS:
        assert tool_name in contracts, f"Missing contract for {tool_name}"


def test_extended_tools_in_global_tools():
    for tool_name in EXTENDED_TOOLS:
        assert tool_name in TOOLS, f"{tool_name} not merged into TOOLS"


def test_seed_tasks_cover_all_domains():
    tasks = generate_seed_tasks(tasks_per_domain=60, seed=42)
    domains = {t.domain for t in tasks}
    expected = set(DOMAIN_TEMPLATES.keys())
    assert domains == expected


def test_split_ratios_for_new_domains():
    tasks = generate_seed_tasks(tasks_per_domain=60, seed=42)
    for domain in ["webhook", "file_storage", "access_control", "notification"]:
        domain_tasks = [t for t in tasks if t.domain == domain]
        assert len(domain_tasks) == 60
        train_count = sum(1 for t in domain_tasks if t.split == "train")
        test_count = sum(1 for t in domain_tasks if t.split == "test")
        assert train_count > 0
        assert test_count > 0


def test_full_episode_for_new_tools():
    env = SimulationEnvironment(EnvironmentConfig())
    policy = DeferFullPolicy()
    tool_configs = {
        "webhook": ("register_webhook", {"url": "https://example.com/hook", "event_type": "push"}, False),
        "file_storage": ("upload_file", {"filename": "test.csv", "size_bytes": 1024, "shared": True}, True),
        "access_control": ("modify_access", {"principal": "user@example.com", "resource": "db", "permission": "write"}, True),
        "notification": ("send_notification", {"message": "Alert", "channel": "slack", "recipients": ["a@b.com"], "deliver_now": True}, True),
    }
    for domain, (tool_name, tool_args, expects_irrev) in tool_configs.items():
        scenario = Scenario(
            scenario_id=f"ext_test_{domain}",
            domain=domain,
            prompt=f"Execute {domain} tool.",
            required_tool=tool_name,
            tool_args=tool_args,
            expects_irreversible=expects_irrev,
            has_delayed_truth=True,
            delayed_truth_category="A",
            delay_mechanism="eventual_consistency",
            metadata={"epsilon": 0.0, "lambda_fault": 0.0, "delayed_truth_category": "A"},
        )
        trace = env.run_episode(
            scenario=scenario,
            policy=policy,
            seed=42,
            epsilon=0.0,
            lambda_fault=0.0,
            repeat_index=0,
        )
        assert trace.episode_id
        assert len(trace.turns) > 0
