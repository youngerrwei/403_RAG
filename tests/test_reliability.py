import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))


class ScriptContractTests(unittest.TestCase):
    """无需真实模型即可执行的部署契约回归。"""

    def test_repository_layout_is_classified(self):
        for directory in (
            "src/lab_rag",
            "scripts",
            "tests",
            "dev",
            "legacy",
            "docs",
            "requirements",
            "data/samples",
        ):
            with self.subTest(directory=directory):
                self.assertTrue((PROJECT_ROOT / directory).is_dir())

        retired_root_files = (
            "rag_agent.py",
            "web_app.py",
            "ingest.py",
            "test_reliability.py",
            "start_rag.sh",
        )
        self.assertFalse([name for name in retired_root_files if (PROJECT_ROOT / name).exists()])

    def test_python_entrypoints_parse(self):
        files = (
            "src/lab_rag/agent_entry.py",
            "src/lab_rag/create_user.py",
            "src/lab_rag/ingest.py",
            "src/lab_rag/logger.py",
            "src/lab_rag/mcp_server.py",
            "src/lab_rag/paths.py",
            "src/lab_rag/rag_agent.py",
            "src/lab_rag/rag_tool.py",
            "src/lab_rag/tools.py",
            "src/lab_rag/web_app.py",
        )
        for file_name in files:
            with self.subTest(file=file_name):
                source = (PROJECT_ROOT / file_name).read_text(encoding="utf-8")
                ast.parse(source, filename=file_name)

    def test_package_has_no_legacy_absolute_internal_imports(self):
        internal_modules = {
            "agent_entry",
            "create_user",
            "ingest",
            "logger",
            "mcp_server",
            "paths",
            "rag_agent",
            "rag_tool",
            "tools",
            "web_app",
        }
        for path in (PROJECT_ROOT / "src" / "lab_rag").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse(
                        node.level == 0 and node.module in internal_modules,
                        f"{path.name}:{node.lineno} 仍使用旧式内部导入: {node.module}",
                    )
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotIn(
                            alias.name,
                            internal_modules,
                            f"{path.name}:{node.lineno} 仍使用旧式内部导入: {alias.name}",
                        )

    def test_shell_scripts_have_valid_bash_syntax(self):
        scripts = [
            "scripts/runtime_common.sh",
            "scripts/setup_env.sh",
            "scripts/download_model.sh",
            "scripts/convert_to_md.sh",
            "scripts/auto_ingest.sh",
            "scripts/start_vllm.sh",
            "scripts/start_rag.sh",
            "scripts/setup_mcp.sh",
            "scripts/start_mcp.sh",
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
            "MCP_CONDA_ENV",
            "MCP_INTERNAL_TOKEN",
            "MCP_INTERNAL_BASE_URL",
            "MCP_HTTP_TIMEOUT",
            "MCP_QUERY_MAX_CHARS",
            "MCP_RESULT_LIMIT",
            "MCP_RESULT_MAX_CHARS",
            "VLLM_STARTUP_TIMEOUT",
            "VLLM_INFERENCE_PROBE_TIMEOUT",
            "VLLM_ENABLE_THINKING",
            "CATALOG_SCAN_LIMIT",
            "ENABLE_TABLE_SEMANTIC_ENRICHMENT",
            "TABLE_SEMANTIC_MAX_ROWS",
            "CONVERT_TABLE_VLM_FALLBACK",
            "TABLE_FLATTENED_LINE_MIN_CHARS",
            "CONVERT_TIMEOUT_PER_MB",
            "CONVERT_MAX_TIMEOUT",
            "WEBAPP_STARTUP_TIMEOUT",
            "HEALTHCHECK_TIMEOUT",
            "RUNTIME_RETRY_INTERVAL",
            "FLASK_SECRET_KEY",
        }
        self.assertFalse(required - values.keys())
        self.assertEqual(values["FLASK_SECRET_KEY"], "")
        self.assertEqual(values["MAX_FILE_SIZE_MB"], "200")
        self.assertEqual(values["CONVERT_TIMEOUT_PER_MB"], "20")
        self.assertEqual(values["CONVERT_MAX_TIMEOUT"], "3600")
        self.assertEqual(values["MCP_INTERNAL_TOKEN"], "")
        self.assertEqual(values["QDRANT_RECREATE_COLLECTION"], "false")

    def test_rag_direct_dependencies_are_declared_and_verified(self):
        requirements = {
            line.strip().lower()
            for line in (PROJECT_ROOT / "requirements" / "rag.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        required_packages = {
            "flask",
            "python-dotenv",
            "langchain",
            "langchain-core",
            "langchain-openai",
            "langchain-huggingface",
            "langchain-qdrant",
            "langchain-text-splitters",
            "qdrant-client",
            "sentence-transformers",
            "tiktoken",
            "modelscope",
            "huggingface-hub",
            "requests",
        }
        self.assertFalse(required_packages - requirements)

        setup_source = (PROJECT_ROOT / "scripts" / "setup_env.sh").read_text(encoding="utf-8")
        self.assertIn("requirements/rag.txt", setup_source)
        for module_name in ("requests", "langchain_text_splitters", "qdrant_client"):
            with self.subTest(module=module_name):
                self.assertIn(module_name, setup_source)

    def test_operational_scripts_expose_hardened_contracts(self):
        converter = (PROJECT_ROOT / "scripts" / "convert_to_md.sh").read_text(encoding="utf-8")
        ingest = (PROJECT_ROOT / "scripts" / "auto_ingest.sh").read_text(encoding="utf-8")
        starter = (PROJECT_ROOT / "scripts" / "start_rag.sh").read_text(encoding="utf-8")
        vllm_starter = (PROJECT_ROOT / "scripts" / "start_vllm.sh").read_text(encoding="utf-8")
        self.assertIn('RAG_ENV_FILE:-${PROJECT_ROOT}/.env', converter)
        self.assertIn("require_option_value", converter)
        self.assertIn("manifest_paths", ingest)
        self.assertIn("PYTHONUNBUFFERED=1", ingest)
        self.assertIn('2>&1 | tee -a "$LOG_FILE"', ingest)
        self.assertIn("PIPESTATUS[0]", ingest)
        self.assertIn("report_lock_holders", ingest)
        self.assertIn('lsof -t "$LOCK_FILE"', ingest)
        self.assertIn("probe_vllm_inference", vllm_starter)
        self.assertIn("真实生成探针失败", vllm_starter)
        self.assertIn("expected_vllm_listener", vllm_starter)
        self.assertIn("stop --orphan", vllm_starter)
        self.assertIn("服务进程启动成功（degraded）", starter)
        self.assertIn("已验证进程身份和健康接口并重新纳管", starter)
        self.assertIn("templates/login.html", starter)
        self.assertIn("templates/index.html", starter)
        self.assertIn("has_suspect_flattened_table", converter)
        self.assertIn("BACKEND=vlm", converter)
        self.assertIn("verify_output_directory_writable", converter)
        self.assertIn("输出目录可创建文件但无法删除", converter)
        self.assertIn("require_positive_integer_config", converter)
        self.assertIn("CONVERT_TIMEOUT_PER_MB", converter)
        self.assertIn("effective_timeout", converter)
        rag_source = (PROJECT_ROOT / "src" / "lab_rag" / "rag_agent.py").read_text(encoding="utf-8")
        self.assertIn("chat_template_kwargs", rag_source)
        self.assertIn("CATALOG_SCAN_LIMIT", rag_source)
        self.assertIn("build_catalog_answer_section", rag_source)
        agent_source = (PROJECT_ROOT / "src" / "lab_rag" / "agent_entry.py").read_text(encoding="utf-8")
        self.assertIn('"tool_name": "list_catalog_entries"', agent_source)
        self.assertIn('rag_result.get("file_result")', agent_source)
        self.assertIn('"file_result": latest_catalog_result', agent_source)

    def test_readme_documents_persistent_cifs_mount(self):
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("//172.18.216.71/Share /mnt/cpu_share cifs", readme)
        self.assertIn("credentials=/etc/samba/cpu_share.credentials", readme)
        self.assertIn("_netdev,nofail,x-systemd.automount", readme)
        self.assertIn("findmnt -rn -t cifs -T /mnt/cpu_share", readme)
        self.assertIn("systemctl start mnt-cpu_share.automount", readme)
        self.assertNotIn("systemctl restart mnt-cpu_share.automount", readme)
        self.assertIn("findmnt -T /mnt/cpu_share", readme)

    def test_shell_entrypoints_target_the_packaged_modules(self):
        entrypoints = {
            "scripts/start_rag.sh": "lab_rag.web_app",
            "scripts/auto_ingest.sh": "lab_rag.ingest",
            "scripts/start_mcp.sh": "lab_rag.mcp_server",
        }
        for script, module in entrypoints.items():
            with self.subTest(script=script):
                source = (PROJECT_ROOT / script).read_text(encoding="utf-8")
                self.assertIn('PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"', source)
                self.assertIn('export PYTHONPATH="$PROJECT_ROOT/src', source)
                self.assertIn(f"-m {module}", source)

        starter = (PROJECT_ROOT / "scripts" / "start_rag.sh").read_text(encoding="utf-8")
        self.assertIn('bash "$SCRIPT_DIR/start_vllm.sh" --background', starter)

    def test_converter_help_and_missing_value_are_runtime_independent(self):
        converter = str(PROJECT_ROOT / "scripts" / "convert_to_md.sh")
        help_result = subprocess.run(
            ["bash", converter, "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--source", help_result.stdout)

        missing_value = subprocess.run(
            ["bash", converter, "--source"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(missing_value.returncode, 0)
        self.assertIn("缺少取值", missing_value.stdout + missing_value.stderr)


class UserFileContractTests(unittest.TestCase):
    def test_malformed_user_file_is_not_silently_reset(self):
        from lab_rag import create_user

        with tempfile.TemporaryDirectory() as temp_dir:
            users_file = Path(temp_dir) / "users.json"
            users_file.write_text("{broken", encoding="utf-8")
            with patch.object(create_user, "USERS_FILE", users_file):
                with self.assertRaises(ValueError):
                    create_user.load_users()
            self.assertEqual(users_file.read_text(encoding="utf-8"), "{broken")

    def test_user_file_save_is_atomic_and_round_trips(self):
        from lab_rag import create_user

        users = [{"username": "tester", "password_hash": "redacted"}]
        with tempfile.TemporaryDirectory() as temp_dir:
            users_file = Path(temp_dir) / "config" / "users.json"
            with patch.object(create_user, "USERS_FILE", users_file):
                create_user.save_users(users)
                self.assertEqual(create_user.load_users(), users)
            self.assertEqual(list(users_file.parent.glob("*.tmp")), [])


class WebContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lab_rag import rag_agent, web_app

        cls.rag_agent = rag_agent
        cls.web_app = web_app

    def _client(self):
        client = self.web_app.app.test_client()
        with client.session_transaction() as session:
            session["logged_in"] = True
            session["username"] = "reliability-test"
        return client

    def test_login_page_renders_from_packaged_template(self):
        self.assertEqual(
            Path(self.web_app.app.template_folder).resolve(),
            (PROJECT_ROOT / "src" / "lab_rag" / "templates").resolve(),
        )
        response = self.web_app.app.test_client().get("/login")
        self.assertEqual(response.status_code, 200)
        self.assertIn("LAB-403 登录", response.get_data(as_text=True))

    def test_vlc_hybrid_route_lists_group_devices(self):
        question = "VLC小组的设备和使用规范"
        route = self.rag_agent.rule_based_route(question)
        self.assertEqual(route["route"], "hybrid")
        self.assertEqual(route["target"], "VLC小组")

        with tempfile.TemporaryDirectory() as temp_dir:
            group_dir = Path(temp_dir) / "设备操作指南" / "VLC小组"
            device_dir = group_dir / "HOLOEYE LETO-3 空间光调制器"
            device_dir.mkdir(parents=True)
            (group_dir / "VLC小组设备使用规范.md").write_text("规范", encoding="utf-8")

            runtime = {
                "config": {
                    "KNOWLEDGE_BASE_ROOT": temp_dir,
                    "ENABLE_FILESYSTEM_TOOL": True,
                    "FILE_SEARCH_LIMIT": 200,
                    "DIRECTORY_CHILD_LIMIT": 200,
                }
            }
            with patch.object(self.rag_agent, "_runtime", runtime):
                result = self.rag_agent.list_catalog_entries(route["target"])

        self.assertEqual(result["mode"], "filesystem_directory")
        self.assertEqual(result["matched_dir"], "设备操作指南/VLC小组")
        self.assertIn("HOLOEYE LETO-3 空间光调制器", {item["name"] for item in result["directories"]})
        self.assertIn("VLC小组设备使用规范.md", {item["name"] for item in result["files"]})
        self.assertEqual(self.rag_agent.catalog_result_count(result), 2)
        catalog_queries = self.rag_agent.build_catalog_retrieval_queries(result, route["target"], question)
        self.assertNotIn(question, catalog_queries)
        self.assertTrue(all("使用方法" in query for query in catalog_queries))

    def test_content_format_routes_to_document_search(self):
        route = self.rag_agent.rule_based_route("文档格式是什么")
        self.assertEqual(route["route"], "rag_search")
        self.assertEqual(route["target"], "文档格式是什么")
        self.assertFalse(self.rag_agent.should_skip_rewrite("规范怎么写"))
        self.assertTrue(self.rag_agent.should_skip_rewrite("文档格式"))

        forbidden_entity = "\u5468\u62a5"
        for module_name in ("rag_agent.py", "agent_entry.py"):
            runtime_source = (PROJECT_ROOT / "src" / "lab_rag" / module_name).read_text(encoding="utf-8")
            self.assertNotIn(forbidden_entity, runtime_source)

        # 明确同时询问设备清单和规范的既有场景仍保持混合路由。
        hybrid = self.rag_agent.rule_based_route("VLC小组有哪些设备及使用规范")
        self.assertEqual(hybrid["route"], "hybrid")

    def test_qdrant_catalog_collapses_manuals_to_device_level(self):
        entries = [
            {
                "rel_path": "设备操作指南/VLC小组/HOLOEYE LETO-3 空间光调制器/Manuals/LETO-3-Manual.md",
                "count": 100,
            },
            {
                "rel_path": "设备操作指南/VLC小组/HOLOEYE LETO-3 空间光调制器/Application.md",
                "count": 20,
            },
            {
                "rel_path": "设备操作指南/VLC小组/Keysight EDU36311A 电源/User-Guide.md",
                "count": 30,
            },
        ]
        result = self.rag_agent.collapse_qdrant_entries_to_directory(entries, "VLC小组")
        self.assertEqual(result["mode"], "qdrant_directory")
        self.assertEqual(result["matched_dir"], "设备操作指南/VLC小组")
        self.assertEqual(
            {item["name"] for item in result["directories"]},
            {"HOLOEYE LETO-3 空间光调制器", "Keysight EDU36311A 电源"},
        )
        self.assertEqual(result["files"], [])

    def test_summary_sanitization_removes_thinking_and_pollution(self):
        self.assertEqual(self.rag_agent.sanitize_summary("<think>internal</think> clean summary"), "clean summary")
        self.assertEqual(self.rag_agent.sanitize_summary("知识库中未找到足够相关内容。"), "")
        leaked = "有效回答。\n```\n上下文结果：\n[目录/文件结果]\n重复内容"
        self.assertEqual(self.rag_agent.sanitize_generated_answer(leaked), "有效回答。")

    def test_retrieval_display_preserves_markdown_table_rows(self):
        table = "| 类型 | 格式 |\n| --- | --- |\n| 向量、矩阵 | 正体+粗体 |"
        cleaned = self.rag_agent.clean_retrieval_display_text(table)
        self.assertEqual(cleaned.count("\n"), 2)
        self.assertIn("| 向量、矩阵 | 正体+粗体 |", cleaned)

    def test_notation_answer_respects_ordered_vector_matrix_examples(self):
        context = (
            "[来源1] [文档标题] 论文格式规范\n"
            "- 变量类型=向量、矩阵；英文字母=正体+粗体，例：$\\mathbf{k}$、$\\mathbf{K}$；"
            "罗马字母=正体+粗体，例：$\\bs{\\lambda}$、$\\bs{\\theta}$"
        )
        answer = self.rag_agent.build_grounded_notation_answer(context)
        self.assertIn(r"向量使用小写正体粗体，如 $\mathbf{k}$", answer)
        self.assertIn(r"矩阵使用大写正体粗体，如 $\mathbf{K}$", answer)
        self.assertNotIn("并未用大小写区分", answer)

    def test_agent_multiline_answer_excludes_reasoning_prefix(self):
        from lab_rag.agent_entry import parse_answer

        output = "Thought: 已完成检索。\nAnswer: 第一行\n第二行\n主要来源：VLC小组"
        self.assertEqual(parse_answer(output), "第一行\n第二行\n主要来源：VLC小组")

    def test_agent_tool_contract_is_route_generic(self):
        from lab_rag.agent_entry import expected_agent_tool

        self.assertEqual(expected_agent_tool("rag_search", False, False), "rag_qa")
        self.assertIsNone(expected_agent_tool("rag_search", False, True))
        self.assertEqual(expected_agent_tool("file_list", False, False), "list_group_files")
        self.assertIsNone(expected_agent_tool("file_list", True, False))
        self.assertEqual(expected_agent_tool("hybrid", False, False), "list_group_files")
        self.assertEqual(expected_agent_tool("hybrid", True, False), "rag_qa")
        self.assertIsNone(expected_agent_tool("hybrid", True, True))

    def test_agent_model_must_emit_the_tool_action(self):
        from lab_rag import agent_entry

        class Response:
            def __init__(self, content):
                self.content = content

        class FakeLlm:
            def __init__(self):
                self.responses = iter((
                    'Thought: 先找模板。\nAction: list_group_files["文档格式"]',
                    'Thought: 应检索正文。\nAction: rag_qa["规范怎么写"]',
                    'Answer: 根据知识库内容回答 [来源1]',
                ))
                self.invoke_count = 0

            def invoke(self, _messages):
                self.invoke_count += 1
                return Response(next(self.responses))

        fake_llm = FakeLlm()
        with patch.object(agent_entry, "build_qwen_llm", return_value=fake_llm), \
                patch.object(
                    agent_entry,
                    "ask_rag",
                    return_value={"answer": "知识库观察结果 [来源1]", "file_result": None},
                ) as mocked_rag, \
                patch.object(agent_entry, "list_catalog_entries") as mocked_catalog, \
                patch.object(agent_entry, "append_user_chat_history"):
            events = list(agent_entry.ask_agent_stream("规范怎么写", username="tester"))

        self.assertEqual(fake_llm.invoke_count, 3)
        mocked_catalog.assert_not_called()
        mocked_rag.assert_called_once_with("规范怎么写", username="tester")
        tool_events = [event for event in events if event["type"] == "step_tool"]
        self.assertEqual([event["tool"] for event in tool_events], ["rag_qa"])
        self.assertEqual(events[-1]["content"], "根据知识库内容回答 [来源1]")

    def test_health_requires_llm_but_allows_qdrant_degraded(self):
        with self._client() as client:
            with patch(
                    "lab_rag.web_app.get_runtime_status",
                    return_value={"embedding": False, "reranker": False, "qdrant": False, "llm": True},
            ):
                response = client.get("/api/health")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json()["status"], "degraded")

            with patch(
                    "lab_rag.web_app.get_runtime_status",
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

        with self._client() as client, patch("lab_rag.web_app.ask_rag_stream", fake_stream):
            response = client.post("/ask_stream", json={"question": "test"}, buffered=True)

        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body.count("data: [DONE]"), 1)
        self.assertEqual(self.web_app._request_semaphore._value, before)

    def test_sse_releases_slot_when_client_disconnects(self):
        before = self.web_app._request_semaphore._value

        def fake_stream(question, username):
            yield {"type": "chunk", "content": "ok"}

        with self._client() as client, patch("lab_rag.web_app.ask_rag_stream", fake_stream):
            response = client.post("/ask_stream", json={"question": "test"}, buffered=False)
            next(response.response)
            response.close()

        self.assertEqual(self.web_app._request_semaphore._value, before)


class IngestContractTests(unittest.TestCase):
    def test_clean_text_preserves_and_enriches_variable_format_table(self):
        from lab_rag.ingest import clean_text

        markdown = """### 3、变量格式：

| 变量类型 | 英文字母 | 罗马字母 |
| --- | --- | --- |
| 标量 | 斜体，例：$k$ | 不限格式，例：$\\alpha$ |
| 集合 | 正体+大写，例：$\\mathrm{K}$ | 大写，例：$\\Omega$、$\\Phi$ |
| 向量、矩阵 | 正体+粗体，例：$\\mathbf{k}$、$\\mathbf{K}$ | 正体+粗体，例：$\\bs{\\lambda}$、$\\bs{\\theta}$ |
"""
        cleaned = clean_text(markdown)
        self.assertIn("\n| 变量类型 | 英文字母 | 罗马字母 |\n", cleaned)
        self.assertIn("[表格结构化转写]", cleaned)
        self.assertIn("变量类型=向量、矩阵", cleaned)
        self.assertIn("英文字母=正体+粗体", cleaned)
        self.assertIn(r"罗马字母=正体+粗体，例：$\bs{\lambda}$、$\bs{\theta}$", cleaned)

        flattened = (
            "3、变量格式：变量类型英文字母罗马字母标量斜体，例：k不限格式，例：α"
            "集合正体+大写，例：K大写，例：Ω、Φ向量、矩阵正体+粗体，例：k、K"
            "正体+粗体，例：λ、θ 注：罗马字母的正体+粗体命令：\\bs{}"
        )
        recovered = clean_text(flattened)
        self.assertIn("[变量格式表语义恢复]", recovered)
        self.assertIn("明确规则：向量和矩阵均使用正体粗体，不能写成普通斜体", recovered)
        self.assertIn(r"英文字母向量用小写正体粗体 $\mathbf{k}$", recovered)
        self.assertIn(r"矩阵用大写正体粗体 $\mathbf{K}$", recovered)

        generic_table = "| 参数 | 单位 |\n| --- | --- |\n| 功率 | dBm |"
        generic_cleaned = clean_text(generic_table)
        self.assertIn("参数=功率；单位=dBm", generic_cleaned)

    def test_vllm_availability_requires_real_generation(self):
        from lab_rag import ingest

        class FakeResponse:
            def __init__(self, status_code=200, payload=None):
                self.status_code = status_code
                self._payload = payload or {}

            def json(self):
                return self._payload

        cfg = {
            "VLLM_BASE_URL": "http://127.0.0.1:8000/v1",
            "VLLM_API_KEY": "test-key",
            "VLLM_MODEL_NAME": "./models/Qwen3-8B-Instruct",
            "VLLM_INFERENCE_PROBE_TIMEOUT": 1,
        }
        with patch.object(ingest.requests, "get", return_value=FakeResponse()):
            with patch.object(
                    ingest.requests,
                    "post",
                    return_value=FakeResponse(payload={"choices": [{"message": {"content": "OK"}}]}),
            ):
                self.assertTrue(ingest.check_vllm_availability(cfg))
            with patch.object(ingest.requests, "post", side_effect=ingest.requests.Timeout):
                self.assertFalse(ingest.check_vllm_availability(cfg))

    def test_parent_ids_are_deterministic(self):
        from lab_rag.ingest import generate_parent_point_id

        item = {"source": "/docs/a.md", "parent_id": "abc"}
        self.assertEqual(generate_parent_point_id(item), generate_parent_point_id(dict(item)))

    def test_stale_delete_happens_only_for_removed_ids(self):
        from lab_rag.ingest import delete_stale_ids

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


class InternalMcpContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lab_rag import web_app

        cls.web_app = web_app

    def _post(self, path, payload, token="test-mcp-token", remote_addr="127.0.0.1"):
        return self.web_app.app.test_client().post(
            path,
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            environ_overrides={"REMOTE_ADDR": remote_addr},
        )

    def test_internal_api_is_disabled_without_token(self):
        with patch("lab_rag.web_app._MCP_INTERNAL_TOKEN", ""):
            response = self._post("/api/internal/mcp/search", {"query": "test"})
        self.assertEqual(response.status_code, 503)

    def test_internal_api_rejects_wrong_token_and_non_loopback_peer(self):
        with patch("lab_rag.web_app._MCP_INTERNAL_TOKEN", "test-mcp-token"):
            wrong_token = self._post(
                "/api/internal/mcp/search", {"query": "test"}, token="wrong"
            )
            remote_peer = self.web_app.app.test_client().post(
                "/api/internal/mcp/search",
                json={"query": "test"},
                headers={
                    "Authorization": "Bearer test-mcp-token",
                    "X-Forwarded-For": "127.0.0.1",
                },
                environ_overrides={"REMOTE_ADDR": "192.0.2.10"},
            )
        self.assertEqual(wrong_token.status_code, 401)
        self.assertEqual(remote_peer.status_code, 403)

    def test_search_reuses_standard_stream_without_answer_or_history(self):
        before = self.web_app._request_semaphore._value
        state = {"after_retrieval": False}
        observed = {}

        def fake_stream(question, username, persist_history):
            observed.update({
                "question": question,
                "username": username,
                "persist_history": persist_history,
            })
            yield {
                "type": "metadata",
                "stage": "route",
                "route": "rag_search",
                "route_target": question,
                "route_reason": "test",
            }
            yield {
                "type": "metadata",
                "stage": "retrieval",
                "rewritten_question": "rewritten",
                "keywords": ["alpha"],
                "queries": ["rewritten"],
                "retrievals": [
                    {
                        "index": 1,
                        "doc_title": "A",
                        "file_name": "a.md",
                        "rel_path": "docs/a.md",
                        "rerank_score": "0.9",
                        "summary": "s" * 2000,
                        "preview": "p" * 2000,
                        "content": "c" * 2000,
                    },
                    {"index": 2, "doc_title": "B", "content": "unused"},
                ],
                "source_map": {"1": {"rel_path": "docs/a.md"}},
            }
            state["after_retrieval"] = True
            raise AssertionError("MCP 不应进入最终答案生成阶段")

        with (
            patch("lab_rag.web_app._MCP_INTERNAL_TOKEN", "test-mcp-token"),
            patch("lab_rag.web_app._MCP_RESULT_MAX_CHARS", 12),
            patch("lab_rag.web_app.ask_rag_stream", fake_stream),
        ):
            response = self._post(
                "/api/internal/mcp/search", {"query": "original", "limit": 1}
            )

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["success"])
        self.assertEqual(observed["question"], "original")
        self.assertEqual(observed["username"], "mcp-bridge")
        self.assertFalse(observed["persist_history"])
        self.assertFalse(state["after_retrieval"])
        self.assertEqual(len(payload["results"]), 1)
        self.assertEqual(len(payload["results"][0]["content"]), 12)
        self.assertEqual(self.web_app._request_semaphore._value, before)

    def test_catalog_is_capped_and_releases_concurrency_slot(self):
        before = self.web_app._request_semaphore._value
        fake_result = {
            "mode": "search",
            "entries": [{"name": "a"}, {"name": "b"}],
        }
        with (
            patch("lab_rag.web_app._MCP_INTERNAL_TOKEN", "test-mcp-token"),
            patch("lab_rag.web_app.list_catalog_entries", return_value=fake_result),
        ):
            response = self._post(
                "/api/internal/mcp/catalog", {"keyword": "paper", "limit": 1}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["catalog"]["entries"], [{"name": "a"}])
        self.assertEqual(self.web_app._request_semaphore._value, before)


class McpBridgeContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_mcp_tools_are_discoverable_and_callable_in_memory(self):
        try:
            from mcp import Client
            from mcp.server import MCPServer  # noqa: F401 - 同时确认使用的是 SDK v2
        except ImportError:
            self.skipTest("MCP SDK v2 仅安装在可选的 rag-mcp 环境")

        from lab_rag import mcp_server

        bridge_result = {"success": True, "route": "rag_search", "results": []}
        with patch.object(mcp_server, "_post_json", return_value=bridge_result):
            async with Client(mcp_server.mcp, raise_exceptions=True) as client:
                tools = await client.list_tools()
                names = {tool.name for tool in tools.tools}
                result = await client.call_tool(
                    "search_lab_knowledge", {"query": "test", "limit": 2}
                )

        self.assertEqual(names, {"search_lab_knowledge", "list_lab_catalog"})
        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content, bridge_result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
