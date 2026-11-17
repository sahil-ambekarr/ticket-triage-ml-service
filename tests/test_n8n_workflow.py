"""Structural checks on the n8n export. This proves the JSON is well-formed and wired as intended;
it does not prove n8n accepts it (the workflow has not been imported into a running n8n)."""

import json
import uuid
from pathlib import Path

import pytest

WORKFLOW = Path(__file__).resolve().parents[1] / "n8n" / "ticket-triage-workflow.json"


@pytest.fixture(scope="module")
def wf() -> dict:
    return json.loads(WORKFLOW.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def nodes(wf) -> dict:
    return {n["name"]: n for n in wf["nodes"]}


def targets(wf: dict, source: str, output: int) -> list[str]:
    return [c["node"] for c in wf["connections"][source]["main"][output]]


def test_top_level_shape(wf):
    assert {"name", "nodes", "connections", "settings"} <= set(wf)
    assert wf["active"] is False


def test_nodes_have_required_fields_and_unique_ids(wf):
    names, ids = set(), set()
    for node in wf["nodes"]:
        assert {"id", "name", "type", "typeVersion", "position", "parameters"} <= set(node)
        assert node["type"].startswith("n8n-nodes-base.")
        assert len(node["position"]) == 2
        uuid.UUID(node["id"])
        names.add(node["name"])
        ids.add(node["id"])
    assert len(names) == len(ids) == len(wf["nodes"])


def test_every_connection_points_at_an_existing_node(wf, nodes):
    for source, outputs in wf["connections"].items():
        assert source in nodes
        for branch in outputs["main"]:
            for link in branch:
                assert link["node"] in nodes and link["type"] == "main"


def test_webhook_to_triage_api_to_if(wf, nodes):
    hook = nodes["Ticket Webhook"]
    assert hook["type"] == "n8n-nodes-base.webhook"
    assert hook["parameters"]["httpMethod"] == "POST"
    assert hook["parameters"]["responseMode"] == "responseNode"
    assert targets(wf, "Ticket Webhook", 0) == ["Call Triage API"]

    api = nodes["Call Triage API"]
    assert api["type"] == "n8n-nodes-base.httpRequest"
    assert api["parameters"]["url"].endswith("/triage")
    assert api["parameters"]["method"] == "POST"
    assert api["retryOnFail"] is True and api["maxTries"] >= 2
    assert targets(wf, "Call Triage API", 0) == ["Needs Human Review?"]


def test_if_branches_on_needs_human_review(wf, nodes):
    cond = nodes["Needs Human Review?"]["parameters"]["conditions"]["conditions"][0]
    assert "needs_human_review" in cond["leftValue"]
    assert targets(wf, "Needs Human Review?", 0) == ["Queue For Human Review"]
    assert targets(wf, "Needs Human Review?", 1) == ["Auto Route To Category Queue"]
    for branch in ("Queue For Human Review", "Auto Route To Category Queue"):
        assert targets(wf, branch, 0) == ["Respond With Triage"]


def test_error_branch_responds_with_502(wf, nodes):
    assert nodes["Call Triage API"]["onError"] == "continueErrorOutput"
    assert targets(wf, "Call Triage API", 1) == ["Build Error Response"]
    assert targets(wf, "Build Error Response", 0) == ["Respond With Error"]
    assert nodes["Respond With Error"]["parameters"]["options"]["responseCode"] == 502


def test_no_credentials_or_pinned_data(wf):
    assert wf.get("pinData", {}) == {}
    assert all("credentials" not in n for n in wf["nodes"])
