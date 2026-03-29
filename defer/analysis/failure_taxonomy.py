from __future__ import annotations

REAL_WORLD_FAILURE_TAXONOMY: dict[str, dict] = {
    "eventual_consistency": {
        "description": (
            "Writes propagate asynchronously across replicas. A read immediately after "
            "a write may return stale data until convergence completes."
        ),
        "real_world_examples": [
            "AWS DynamoDB eventual consistency reads (AWS docs)",
            "Cassandra read-after-write with quorum < ALL",
            "Google Cloud Spanner external consistency lag under cross-region writes",
        ],
        "mapped_delay_mechanism": "eventual_consistency",
        "severity": "high",
    },
    "async_job_completion": {
        "description": (
            "Long-running background tasks (e.g., data exports, ML training jobs) "
            "return a job ID immediately but complete minutes to hours later."
        ),
        "real_world_examples": [
            "Stripe asynchronous payout completion (Stripe API docs)",
            "AWS Batch job lifecycle: SUBMITTED -> RUNNABLE -> SUCCEEDED/FAILED",
            "Google BigQuery async query jobs with polling for results",
        ],
        "mapped_delay_mechanism": "async_job_completion",
        "severity": "medium",
    },
    "concurrent_edit_conflict": {
        "description": (
            "Multiple agents or users modify the same resource simultaneously, "
            "causing last-write-wins overwrites or merge conflicts."
        ),
        "real_world_examples": [
            "Git merge conflicts in collaborative development",
            "Google Docs concurrent edit resolution (OT/CRDT)",
            "Netflix microservice data races during canary deployments (Netflix tech blog)",
        ],
        "mapped_delay_mechanism": "concurrent_edit_conflict",
        "severity": "critical",
    },
    "stale_schema_cache": {
        "description": (
            "Cached schema or configuration becomes stale after a migration or "
            "config change, causing tools to operate on outdated structure."
        ),
        "real_world_examples": [
            "PostgreSQL prepared statement cache invalidation after ALTER TABLE",
            "AWS API Gateway caching stale API definitions after deployment",
            "Kubernetes ConfigMap propagation delay to running pods",
        ],
        "mapped_delay_mechanism": "stale_schema_cache",
        "severity": "high",
    },
    "delayed_authorization": {
        "description": (
            "Permission or authorization changes propagate with delay through "
            "IAM systems, causing temporary access denials or stale grants."
        ),
        "real_world_examples": [
            "AWS IAM policy propagation delay (up to several seconds, AWS docs)",
            "Google Cloud IAM eventual consistency for policy bindings",
            "Azure RBAC role assignment propagation (up to 5 minutes, Azure docs)",
        ],
        "mapped_delay_mechanism": "delayed_authorization",
        "severity": "critical",
    },
    "cross_tool_evidence_lag": {
        "description": (
            "Evidence gathered from one tool becomes stale or contradicted by "
            "data from another tool due to unsynchronized data sources."
        ),
        "real_world_examples": [
            "Monitoring dashboards (Grafana/Datadog) showing different values than direct DB queries",
            "Slack notification timestamps diverging from actual event times in incident response",
            "CDN cache serving stale content while origin has been updated",
        ],
        "mapped_delay_mechanism": "cross_tool_evidence_lag",
        "severity": "high",
    },
}


def validate_taxonomy_coverage(delay_mechanisms: list[str]) -> dict[str, bool]:
    return {
        mechanism: mechanism in REAL_WORLD_FAILURE_TAXONOMY
        for mechanism in delay_mechanisms
    }


def taxonomy_to_latex_table() -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Real-World Failure Taxonomy}",
        r"\label{tab:failure-taxonomy}",
        r"\begin{tabular}{llp{5cm}l}",
        r"\toprule",
        r"Delay Mechanism & Severity & Description & Example \\",
        r"\midrule",
    ]
    for mechanism, entry in sorted(REAL_WORLD_FAILURE_TAXONOMY.items()):
        mechanism_tex = mechanism.replace("_", "\\_")
        desc = entry["description"][:80] + "..."
        example = entry["real_world_examples"][0] if entry["real_world_examples"] else "N/A"
        lines.append(
            f"  {mechanism_tex} & {entry['severity']} & {desc} & {example} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)
