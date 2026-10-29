import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
import yaml

CHART_PATH = Path(__file__).resolve().parents[3] / "charts" / "model-engine"
VALUES_PATH = CHART_PATH / "values_circleci.yaml"

RELEASE_NAMESPACE = "launch"
ENDPOINT_NAMESPACE = "launch-inference"
SERVICE_ACCOUNT_NAME = "model-engine"
# (namespace, name) of the accounts scoped RBAC distinguishes. Endpoint pods run as the chart's
# account in the endpoint namespace when serviceTemplate.serviceAccountName is the fullname.
CONTROL_PLANE = (RELEASE_NAMESPACE, SERVICE_ACCOUNT_NAME)
MODEL_PODS = (ENDPOINT_NAMESPACE, SERVICE_ACCOUNT_NAME)
ACCOUNTS = [CONTROL_PLANE, MODEL_PODS]
ACCOUNT_IDS = ["control-plane", "model-pods"]
SPLIT_NAMESPACE_ARGS = [
    "--namespace",
    RELEASE_NAMESPACE,
    "--set",
    f"serviceAccount.namespaces={{{RELEASE_NAMESPACE},{ENDPOINT_NAMESPACE}}}",
    "--set",
    f"config.values.launch.endpoint_namespace={ENDPOINT_NAMESPACE}",
]
SCOPED_ARGS = ["--set", "rbac.clusterAdmin=false"]
DEFAULT_LISTED_ARGS = [
    "--set",
    f"serviceAccount.namespaces={{{RELEASE_NAMESPACE},{ENDPOINT_NAMESPACE},default}}",
]
PREWARMING_ARGS = ["--set", "autoscaling.prewarming.enabled=true"]
RESTART_KEDA_ARGS = ["--set", "restartKedaOperator=true"]

# (verb, API group, resource) for every call model-engine makes in the endpoint namespace.
ENDPOINT_NAMESPACE_CALLS = [
    *[(verb, "", "configmaps") for verb in ["list", "create", "patch", "delete"]],
    *[(verb, "", "services") for verb in ["create", "patch", "delete"]],
    *[(verb, "", "persistentvolumeclaims") for verb in ["create", "delete"]],
    ("list", "", "pods"),
    *[
        (verb, "apps", "deployments")
        for verb in ["get", "list", "create", "update", "patch", "delete"]
    ],
    *[(verb, "apps", "daemonsets") for verb in ["list", "create", "update"]],
    *[
        (verb, "autoscaling", "horizontalpodautoscalers")
        for verb in ["get", "list", "create", "patch", "delete"]
    ],
    *[(verb, "policy", "poddisruptionbudgets") for verb in ["get", "create", "update", "delete"]],
    *[(verb, "batch", "jobs") for verb in ["list", "create", "update", "delete"]],
    *[(verb, "batch", "cronjobs") for verb in ["create", "patch", "delete"]],
    *[
        (verb, group, resource)
        for group, resource in [
            ("autoscaling.k8s.io", "verticalpodautoscalers"),
            ("keda.sh", "scaledobjects"),
            ("leaderworkerset.x-k8s.io", "leaderworkersets"),
        ]
        for verb in ["get", "list", "create", "update", "delete"]
    ],
    *[
        (verb, "networking.istio.io", resource)
        for resource in ["destinationrules", "virtualservices", "serviceentries"]
        for verb in ["get", "create", "update", "delete"]
    ],
]
# ECR image builds (`kubectl apply`, pod watch, `kubectl logs`) and the celery autoscaler's scan.
DEFAULT_NAMESPACE_CALLS = [
    *[(verb, "batch", "jobs") for verb in ["get", "create", "patch"]],
    *[(verb, "", "pods") for verb in ["get", "list", "watch"]],
    ("get", "", "pods/log"),
    *[(verb, "apps", "deployments") for verb in ["get", "list", "patch"]],
]
WORKLOAD_CREATES = [
    ("create", "apps", "deployments"),
    ("create", "batch", "jobs"),
    ("create", "batch", "cronjobs"),
    ("create", "", "pods"),
]


def _render(extra_args: List[str], templates: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if shutil.which("helm") is None:
        pytest.skip("helm is not installed")

    command = ["helm", "template", "test-release", str(CHART_PATH), "-f", str(VALUES_PATH)]
    for template in templates or []:
        command.extend(["--show-only", template])
    command.extend(extra_args)
    rendered = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    return [doc for doc in yaml.safe_load_all(rendered) if doc]


def _render_rbac(extra_args: List[str]) -> List[Dict[str, Any]]:
    return _render(SPLIT_NAMESPACE_ARGS + SCOPED_ARGS + extra_args, ["templates/rbac.yaml"])


def _matches(allowed: List[str], value: str) -> bool:
    return "*" in allowed or value in allowed


def _can_i(
    docs: List[Dict[str, Any]],
    service_account_namespace: str,
    verb: str,
    group: str,
    resource: str,
    namespace: Optional[str],
    name: Optional[str] = None,
    service_account_name: str = SERVICE_ACCOUNT_NAME,
) -> bool:
    roles = {
        (doc["kind"], doc["metadata"].get("namespace"), doc["metadata"]["name"]): doc
        for doc in docs
        if doc["kind"] in ("Role", "ClusterRole")
    }
    subject = {
        "kind": "ServiceAccount",
        "name": service_account_name,
        "namespace": service_account_namespace,
    }
    for binding in docs:
        if binding["kind"] not in ("RoleBinding", "ClusterRoleBinding"):
            continue
        binding_namespace = binding["metadata"].get("namespace")
        if binding["kind"] == "RoleBinding" and binding_namespace != namespace:
            continue
        if subject not in binding["subjects"]:
            continue
        ref = binding["roleRef"]
        role_namespace = binding_namespace if ref["kind"] == "Role" else None
        if (ref["kind"], ref["name"]) == ("ClusterRole", "cluster-admin"):
            return True
        role = roles.get((ref["kind"], role_namespace, ref["name"]))
        assert role is not None, f"{binding['metadata']['name']} references missing {ref}"
        for rule in role["rules"]:
            if (
                _matches(rule["apiGroups"], group)
                and _matches(rule["resources"], resource)
                and _matches(rule["verbs"], verb)
                and (not rule.get("resourceNames") or name in rule["resourceNames"])
            ):
                return True
    return False


def test_default_binds_cluster_admin_in_every_service_account_namespace():
    docs = _render(SPLIT_NAMESPACE_ARGS + PREWARMING_ARGS + RESTART_KEDA_ARGS)

    rbac = [doc for doc in docs if doc["kind"].endswith(("Role", "RoleBinding"))]
    assert [(doc["kind"], doc["roleRef"]["name"]) for doc in rbac] == [
        ("ClusterRoleBinding", "cluster-admin")
    ]
    assert rbac[0]["subjects"] == [
        {"kind": "ServiceAccount", "name": SERVICE_ACCOUNT_NAME, "namespace": namespace}
        for namespace in [RELEASE_NAMESPACE, RELEASE_NAMESPACE, ENDPOINT_NAMESPACE]
    ]


# `helm upgrade --reuse-values` renders with the previous release's values, which may lack `rbac`.
@pytest.mark.parametrize(
    "extra_args",
    [["--set", "rbac=null"], ["--set-string", "rbac.clusterAdmin=false"]],
    ids=["missing", "string-false"],
)
def test_only_boolean_false_turns_off_cluster_admin(extra_args: List[str]):
    docs = _render(SPLIT_NAMESPACE_ARGS + extra_args)

    rbac = [doc for doc in docs if doc["kind"].endswith(("Role", "RoleBinding"))]
    assert [(doc["kind"], doc["roleRef"]["name"]) for doc in rbac] == [
        ("ClusterRoleBinding", "cluster-admin")
    ]


# With config.file the service config lives in the image, so the chart can't see its namespace.
CONFIG_FILE_ARGS = [
    "--set",
    "config.values=null",
    "--set",
    "config.file.launch=/workspace/service_config.yaml",
    "--set",
    "config.file.infra=/workspace/infra_config.yaml",
]


def test_scoped_rbac_requires_an_endpoint_namespace():
    with pytest.raises(subprocess.CalledProcessError) as error:
        _render_rbac(CONFIG_FILE_ARGS)

    assert "needs the endpoint namespace" in error.value.stderr


def test_rbac_endpoint_namespace_overrides_config():
    docs = _render_rbac(CONFIG_FILE_ARGS + ["--set", "rbac.endpointNamespace=other-endpoints"])

    assert {doc["metadata"]["namespace"] for doc in docs} == {RELEASE_NAMESPACE, "other-endpoints"}
    denied = [
        call
        for call in ENDPOINT_NAMESPACE_CALLS
        if not _can_i(docs, RELEASE_NAMESPACE, *call, "other-endpoints")
    ]
    assert denied == []
    assert not _can_i(docs, RELEASE_NAMESPACE, "create", "apps", "deployments", ENDPOINT_NAMESPACE)


def test_scoped_rbac_grants_no_cluster_admin_or_wildcards():
    docs = _render(SPLIT_NAMESPACE_ARGS + SCOPED_ARGS + PREWARMING_ARGS + RESTART_KEDA_ARGS)

    bindings = [doc for doc in docs if doc["kind"] in ("RoleBinding", "ClusterRoleBinding")]
    assert bindings
    for binding in bindings:
        assert binding["roleRef"]["name"] != "cluster-admin"
    for role in (doc for doc in docs if doc["kind"] in ("Role", "ClusterRole")):
        for rule in role["rules"]:
            assert "*" not in rule["apiGroups"] + rule["resources"] + rule["verbs"]


@pytest.mark.parametrize("account", ACCOUNTS, ids=ACCOUNT_IDS)
@pytest.mark.parametrize("verb", ["get", "list", "watch"])
def test_cannot_read_secrets_outside_model_engine_namespaces(account: Tuple[str, str], verb: str):
    docs = _render_rbac(PREWARMING_ARGS + RESTART_KEDA_ARGS)
    namespace, name = account

    for secret_namespace in ["other-team", None]:
        assert not _can_i(
            docs, namespace, verb, "", "secrets", secret_namespace, "signing-key", name
        )


def test_secret_access_is_limited_to_the_codeartifact_secret():
    docs = _render_rbac(PREWARMING_ARGS + RESTART_KEDA_ARGS)

    secret_rules = [
        (role["metadata"].get("namespace"), rule)
        for role in docs
        if role["kind"] in ("Role", "ClusterRole")
        for rule in role["rules"]
        if "secrets" in rule["resources"]
    ]
    assert secret_rules == [
        (
            RELEASE_NAMESPACE,
            {
                "apiGroups": [""],
                "resources": ["secrets"],
                "resourceNames": ["codeartifact-pip-conf"],
                "verbs": ["get", "patch"],
            },
        )
    ]


def test_endpoint_namespace_calls_are_allowed():
    docs = _render_rbac([])

    denied = [
        call
        for call in ENDPOINT_NAMESPACE_CALLS
        if not _can_i(docs, RELEASE_NAMESPACE, *call, ENDPOINT_NAMESPACE)
    ]
    assert denied == []


def test_endpoint_namespace_gets_a_role_when_not_listed_in_service_account_namespaces():
    docs = _render_rbac(["--set", "serviceAccount.namespaces=null"])

    assert [(doc["kind"], doc["metadata"]["namespace"]) for doc in docs] == [
        ("Role", RELEASE_NAMESPACE),
        ("Role", ENDPOINT_NAMESPACE),
        ("RoleBinding", RELEASE_NAMESPACE),
        ("RoleBinding", ENDPOINT_NAMESPACE),
    ]
    denied = [
        call
        for call in ENDPOINT_NAMESPACE_CALLS
        if not _can_i(docs, RELEASE_NAMESPACE, *call, ENDPOINT_NAMESPACE)
    ]
    assert denied == []


def test_release_namespace_config_maps_are_readable():
    docs = _render_rbac([])

    assert _can_i(docs, RELEASE_NAMESPACE, "get", "", "configmaps", RELEASE_NAMESPACE)


# A pod that can be created in a namespace can mount any Secret there and run as any of its
# ServiceAccounts, so the release namespace must not allow workload creation.
@pytest.mark.parametrize("account", ACCOUNTS, ids=ACCOUNT_IDS)
def test_release_namespace_does_not_allow_workloads(account: Tuple[str, str]):
    docs = _render_rbac(PREWARMING_ARGS)
    namespace, name = account

    allowed = [
        call
        for call in WORKLOAD_CREATES + [("patch", "apps", "deployments")]
        if _can_i(docs, namespace, *call, RELEASE_NAMESPACE, None, name)
    ]
    assert allowed == []


def _granted(
    docs: List[Dict[str, Any]], account: Tuple[str, str], namespaces: List[str]
) -> List[Tuple[str, Tuple[str, str, str]]]:
    calls = ENDPOINT_NAMESPACE_CALLS + DEFAULT_NAMESPACE_CALLS + [("get", "", "configmaps")]
    return [
        (namespace, call)
        for namespace in namespaces
        for call in calls
        if _can_i(docs, account[0], *call, namespace, None, account[1])
    ]


# Endpoint pods and docker-image batch jobs run user code, and the endpoint namespace is listed
# in SPLIT_NAMESPACE_ARGS, so the chart creates their account there.
def test_model_pod_account_has_no_access():
    docs = _render_rbac(PREWARMING_ARGS + DEFAULT_LISTED_ARGS)

    assert _granted(docs, MODEL_PODS, [ENDPOINT_NAMESPACE, RELEASE_NAMESPACE, "default"]) == []
    assert not _can_i(
        docs, ENDPOINT_NAMESPACE, "get", "", "secrets", RELEASE_NAMESPACE, "codeartifact-pip-conf"
    )
    assert not _can_i(docs, ENDPOINT_NAMESPACE, "list", "", "nodes", None)


@pytest.mark.parametrize(
    "extra_args, release_namespace",
    [(DEFAULT_LISTED_ARGS, RELEASE_NAMESPACE), (["--namespace", "default"], "default")],
    ids=["default-listed", "default-is-release-namespace"],
)
def test_default_namespace_allows_image_builds_and_autoscaler_scan(
    extra_args: List[str], release_namespace: str
):
    docs = _render_rbac(extra_args)

    denied = [
        call
        for call in DEFAULT_NAMESPACE_CALLS
        if not _can_i(docs, release_namespace, *call, "default")
    ]
    assert denied == []
    # Image builds need Jobs there; nothing needs any other workload.
    allowed = [
        call
        for call in WORKLOAD_CREATES
        if call != ("create", "batch", "jobs") and _can_i(docs, release_namespace, *call, "default")
    ]
    assert allowed == []


def test_other_listed_namespaces_get_no_role():
    docs = _render_rbac(
        ["--set", f"serviceAccount.namespaces={{{RELEASE_NAMESPACE},{ENDPOINT_NAMESPACE},other}}"]
    )

    assert {doc["metadata"]["namespace"] for doc in docs} == {RELEASE_NAMESPACE, ENDPOINT_NAMESPACE}


def test_proportional_autoscaler_rbac_only_when_prewarming():
    without = _render_rbac([])
    assert not _can_i(without, RELEASE_NAMESPACE, "list", "", "nodes", None)
    assert not _can_i(without, RELEASE_NAMESPACE, "create", "", "configmaps", RELEASE_NAMESPACE)

    docs = _render_rbac(PREWARMING_ARGS)
    for verb in ["list", "watch"]:
        assert _can_i(docs, RELEASE_NAMESPACE, verb, "", "nodes", None)
    for verb in ["get", "update"]:
        assert _can_i(docs, RELEASE_NAMESPACE, verb, "apps", "deployments/scale", RELEASE_NAMESPACE)
    for verb in ["get", "create"]:
        assert _can_i(docs, RELEASE_NAMESPACE, verb, "", "configmaps", RELEASE_NAMESPACE)
    (cluster_role,) = [doc for doc in docs if doc["kind"] == "ClusterRole"]
    assert cluster_role["rules"] == [
        {"apiGroups": [""], "resources": ["nodes"], "verbs": ["list", "watch"]}
    ]

    with_service_identifier = _render_rbac(PREWARMING_ARGS + ["--set", "serviceIdentifier=foo"])
    assert not any(doc["kind"] == "ClusterRole" for doc in with_service_identifier)


def test_restart_keda_operator_rbac_only_when_enabled():
    assert not _can_i(
        _render_rbac([]),
        RELEASE_NAMESPACE,
        "patch",
        "apps",
        "deployments",
        "kube-system",
        "keda-operator",
    )

    docs = _render_rbac(RESTART_KEDA_ARGS)
    for verb in ["get", "patch"]:
        assert _can_i(
            docs, RELEASE_NAMESPACE, verb, "apps", "deployments", "kube-system", "keda-operator"
        )
    assert not _can_i(
        docs, RELEASE_NAMESPACE, "patch", "apps", "deployments", "kube-system", "coredns"
    )
