import ast
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent


class ScriptContractTests(unittest.TestCase):
    def test_python_entrypoints_parse(self):
        files = (
            "agent_entry.py",
            "create_user.py",
            "ingest.py",
            "logger.py",
            "rag_agent.py",
            "rag_tool.py",
            "tools.py",
            "web_app.py",
        )
        for file_name in files:
            with self.subTest(file=file_name):
                source = (PROJECT_ROOT / file_name).read_text(encoding="utf-8")
                ast.parse(source, filename=file_name)

    def test_shell_scripts_have_valid_bash_syntax(self):
        scripts = [
            "scripts/runtime_common.sh",
            "setup_env.sh",
            "download_model.sh",
            "convert_to_md.sh",
            "auto_ingest.sh",
            "start_vllm.sh",
            "start_rag.sh",
        ]
        for script in scripts:
            with self.subTest(script=script):
                result = subprocess.run(
                    ["bash", "-n", str(PROJECT_ROOT / script)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_model_name_matching_accepts_absolute_runtime_path(self):
        script = (
            "source scripts/runtime_common.sh; "
            "payload='{\"data\":[{\"id\":\"/srv/models/Qwen3-8B-Instruct/\","
            "\"permission\":[{\"id\":\"modelperm-not-the-model\"}]}]}'; "
            "actual=\"$(printf '%s' \"$payload\" | extract_vllm_model_id \"$1\")\"; "
            "[[ \"$actual\" == '/srv/models/Qwen3-8B-Instruct/' ]] && "
            "model_names_match '/srv/models/Qwen3-8B-Instruct/' "
            "'./models/Qwen3-8B-Instruct' && "
            "! model_names_match '/srv/models/Other' './models/Qwen3-8B-Instruct'"
        )
        result = subprocess.run(
            ["bash", "-c", script, "_", sys.executable],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_public_env_template_is_safe_and_complete(self):
        values = {}
        for raw_line in (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key] = value

        required = {
            "RAG_CONDA_ENV",
            "VLLM_CONDA_ENV",
            "MINERU_CONDA_ENV",
            "VLLM_STARTUP_TIMEOUT",
            "WEBAPP_STARTUP_TIMEOUT",
            "HEALTHCHECK_TIMEOUT",
            "RUNTIME_RETRY_INTERVAL",
            "FLASK_SECRET_KEY",
        }
        self.assertFalse(required - values.keys())
        self.assertEqual(values["FLASK_SECRET_KEY"], "")
        self.assertEqual(values["QDRANT_RECREATE_COLLECTION"], "false")


class WebContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import rag_agent
        import web_app

        cls.rag_agent = rag_agent
        cls.web_app = web_app

    def _client(self):
        client = self.web_app.app.test_client()
        with client.session_transaction() as session:
            session["logged_in"] = True
            session["username"] = "reliability-test"
        return client

    def test_health_requires_llm_but_allows_qdrant_degraded(self):
        with self._client() as client:
            with patch(
                    "web_app.get_runtime_status",
                    return_value={"embedding": False, "reranker": False, "qdrant": False, "llm": True},
            ):
                response = client.get("/api/health")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json()["status"], "degraded")

            with patch(
                    "web_app.get_runtime_status",
                    return_value={"embedding": True, "reranker": True, "qdrant": True, "llm": False},
            ):
                response = client.get("/api/health")
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.get_json()["status"], "error")

    def test_qdrant_health_requires_both_collections(self):
        class FakeResponse:
            status = 200

            def __init__(self, names):
                self._names = names

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps({
                    "result": {"collections": [{"name": name} for name in self._names]}
                }).encode("utf-8")

        required = ("lab_knowledge_base", "lab_knowledge_base_parents")
        checker = self.rag_agent._qdrant_collections_ready
        with patch.object(self.rag_agent.urllib.request, "urlopen", return_value=FakeResponse(required)):
            self.assertTrue(checker("127.0.0.1", 6333, required, 1))
        with patch.object(self.rag_agent.urllib.request, "urlopen", return_value=FakeResponse(required[:1])):
            self.assertFalse(checker("127.0.0.1", 6333, required, 1))

    def test_sse_releases_exactly_one_concurrency_slot(self):
        before = self.web_app._request_semaphore._value

        def fake_stream(question, username):
            yield {"type": "chunk", "content": "ok"}
            yield {"type": "final", "content": "ok"}

        with self._client() as client, patch("web_app.ask_rag_stream", fake_stream):
            response = client.post("/ask_stream", json={"question": "test"}, buffered=True)

        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body.count("data: [DONE]"), 1)
        self.assertEqual(self.web_app._request_semaphore._value, before)

    def test_sse_releases_slot_when_client_disconnects(self):
        before = self.web_app._request_semaphore._value

        def fake_stream(question, username):
            yield {"type": "chunk", "content": "ok"}

        with self._client() as client, patch("web_app.ask_rag_stream", fake_stream):
            response = client.post("/ask_stream", json={"question": "test"}, buffered=False)
            next(response.response)
            response.close()

        self.assertEqual(self.web_app._request_semaphore._value, before)


class IngestContractTests(unittest.TestCase):
    def test_parent_ids_are_deterministic(self):
        from ingest import generate_parent_point_id

        item = {"source": "/docs/a.md", "parent_id": "abc"}
        self.assertEqual(generate_parent_point_id(item), generate_parent_point_id(dict(item)))

    def test_stale_delete_happens_only_for_removed_ids(self):
        from ingest import delete_stale_ids

        class FakeClient:
            def __init__(self):
                self.calls = []

            def delete(self, **kwargs):
                self.calls.append(kwargs)

        client = FakeClient()
        delete_stale_ids(
            client,
            "children",
            {"a": {"keep", "remove"}, "b": {"old"}},
            {"a": {"keep", "new"}, "b": set()},
        )
        self.assertEqual(len(client.calls), 1)
        selector = client.calls[0]["points_selector"]
        self.assertEqual(set(selector.points), {"remove", "old"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
