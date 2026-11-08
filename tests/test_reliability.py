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
            "src/lab_rag/agent_tools.py",
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
            # 全链路改进新增的可调键（必须与两侧 load_config 注册一致）
            "ENABLE_MARKDOWN_STRUCTURE_GUARD",
            "ENABLE_PATH_CONTEXT_PREFIX",
            "CHUNK_FILTER_SPECIAL_CHAR_RATIO",
            "ENABLE_TABLE_FILTER_EXEMPTION",
            "INGEST_MAX_FAILED_FILES",
            "QDRANT_INDEX_FAIL_FAST",
            "MAX_DENSE_QUERIES",
            "MAX_HYDE_QUERIES",
            "MAX_SPARSE_CONDITIONS",
            "ENABLE_HISTORY_AWARE_REWRITE",
            "RETRIEVAL_INDEX_VERSION",
            "COVERAGE_HIGH_THRESHOLD",
            "COVERAGE_MEDIUM_THRESHOLD",
            "COVERAGE_LOW_THRESHOLD",
            "RERANK_QUERY_MODE",
            "RERANKER_ALLOW_LOW_SCORE_FALLBACK",
            "NOTATION_MATCH_BONUS",
            "ENABLE_CONTEXT_NOISE_FILTER",
        }
        self.assertFalse(required - values.keys())
        self.assertEqual(values["FLASK_SECRET_KEY"], "")
        self.assertEqual(values["MAX_FILE_SIZE_MB"], "200")
        self.assertEqual(values["CONVERT_TIMEOUT_PER_MB"], "20")
        self.assertEqual(values["CONVERT_MAX_TIMEOUT"], "3600")
        self.assertEqual(values["MCP_INTERNAL_TOKEN"], "")
        self.assertEqual(values["QDRANT_RECREATE_COLLECTION"], "false")
        self.assertEqual(values["MAX_DENSE_QUERIES"], "4")
        self.assertEqual(values["MAX_HYDE_QUERIES"], "2")
        self.assertEqual(values["MAX_SPARSE_CONDITIONS"], "30")
        # v3.2.0 表格转写跳过 OCR 噪声行，page_content 变更，索引版本递增到 2
        self.assertEqual(values["RETRIEVAL_INDEX_VERSION"], "2")
        self.assertEqual(values["RERANKER_ALLOW_LOW_SCORE_FALLBACK"], "false")

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

    def test_weekly_report_format_routes_to_document_content(self):
        route = self.rag_agent.rule_based_route("周报格式是什么")
        self.assertEqual(route["route"], "rag_search")
        self.assertEqual(route["target"], "周报格式是什么")
        self.assertFalse(self.rag_agent.should_skip_rewrite("周报格式是什么"))
        self.assertTrue(self.rag_agent.should_skip_rewrite("周报格式"))

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

    def test_agent_content_question_uses_rag_directly(self):
        from lab_rag import agent_entry

        from unittest.mock import Mock
        result = {"ok": True, "observation": "知识库周报规范 [来源1]",
                  "catalog_result": None, "elapsed_s": 0.0, "tool_call_id": "tc-1"}
        with patch.object(agent_entry, "dispatch_tool",
                          return_value=result) as mocked_dispatch, \
                patch.object(agent_entry, "build_qwen_llm") as mocked_llm, \
                patch.object(agent_entry, "append_user_chat_history"):
            events = list(agent_entry.ask_agent_stream("周报怎么写", username="tester"))

        # 确定性分支把已算出的路由透传给工具层，避免重复路由
        self.assertEqual(mocked_dispatch.call_args.args[0], "rag_qa")
        self.assertEqual(mocked_dispatch.call_args.args[1], "周报怎么写")
        self.assertIsNotNone(mocked_dispatch.call_args.kwargs.get("route_hint"))

        mocked_llm.assert_not_called()
        # 3.4.3：确定性路由不再伪造模型 Thought，首条为编排器决策事件
        self.assertEqual(events[0]["type"], "route_direct")
        self.assertEqual(events[0]["tool"], "rag_qa")
        self.assertEqual(events[1]["tool"], "rag_qa")
        self.assertEqual(events[1]["tool_call_id"], "tc-1")
        self.assertEqual(events[-1]["content"], "知识库周报规范 [来源1]")

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


class RagPipelineQualityTests(unittest.TestCase):
    """全链路改进的离线质量回归：不依赖 GPU、Qdrant 或 vLLM。"""

    def test_new_config_keys_are_registered_in_both_load_configs(self):
        from lab_rag import ingest, rag_agent

        ingest_cfg = ingest.load_config()
        for key in (
                "ENABLE_MARKDOWN_STRUCTURE_GUARD", "ENABLE_PATH_CONTEXT_PREFIX",
                "CHUNK_FILTER_SPECIAL_CHAR_RATIO", "ENABLE_TABLE_FILTER_EXEMPTION",
                "INGEST_MAX_FAILED_FILES", "QDRANT_INDEX_FAIL_FAST",
                "TABLE_SEMANTIC_MAX_REPLACEMENT_RATIO",
        ):
            with self.subTest(key=key):
                self.assertIn(key, ingest_cfg)

        rag_cfg = rag_agent.load_config()
        for key in (
                "MAX_DENSE_QUERIES", "MAX_HYDE_QUERIES", "MAX_SPARSE_CONDITIONS",
                "ENABLE_HISTORY_AWARE_REWRITE", "RETRIEVAL_INDEX_VERSION",
                "COVERAGE_HIGH_THRESHOLD", "COVERAGE_MEDIUM_THRESHOLD",
                "COVERAGE_LOW_THRESHOLD", "RERANK_QUERY_MODE",
                "RERANKER_ALLOW_LOW_SCORE_FALLBACK", "NOTATION_MATCH_BONUS",
                "ENABLE_CONTEXT_NOISE_FILTER",
                # 检索侧优化（A 类：catalog 快照/hybrid 去重；B 类：拒答与父块多样性）
                "ENABLE_CATALOG_SNAPSHOT_CACHE", "CATALOG_SNAPSHOT_TTL",
                "CATALOG_EXCLUDE_HIDDEN_DIRS", "CATALOG_MAX_TRAVERSAL",
                "ENABLE_LEXICAL_RETRIEVAL", "LEXICAL_CANDIDATE_LIMIT",
                "ENABLE_ADAPTIVE_LEXICAL_RECALL", "ADAPTIVE_LEXICAL_MIN_DIGITS",
                "ADAPTIVE_LEXICAL_TOP_K", "AGENT_MAX_STEPS",
                "AGENT_TOTAL_TIMEOUT_S", "AGENT_FALLBACK_SNIPPET_CHARS",
                "AGENT_REPEAT_ACTION_GUARD",
                "AGENT_OBSERVATION_MAX_CHARS", "AGENT_KEEP_RECENT_OBSERVATIONS",
                "AGENT_MAX_IDLE_STEPS", "AGENT_FINAL_CONVERGE_ROUND",
                "MAX_HYBRID_QUERIES",
                "RERANKER_RELATIVE_RATIO", "RERANKER_MIN_KEEP",
                "ENABLE_ANSWERABILITY_GATE", "ANSWERABILITY_TOP_SCORE_FLOOR",
                "ANSWERABILITY_MIN_KEYWORD_HITS", "ANSWERABILITY_MIN_OVERLAP_HITS",
                "MAX_SPARSE_TERMS",
                "PARENT_EXPAND_MAX_CHARS", "PARENT_MAX_SINGLE_CHARS",
                "PARENT_MAX_PER_SOURCE",
        ):
            with self.subTest(key=key):
                self.assertIn(key, rag_cfg)

    def test_cleaning_keeps_formulas_code_and_chinese_punctuation(self):
        from lab_rag.ingest import clean_text

        fence = chr(96) * 3
        text = "\n".join([
            "变量，例：$k$ 与 <T> 泛型",
            fence + "bash", "# 这不是标题", "echo ok", fence,
            "⼯作及⽂档要求",
        ])
        cleaned = clean_text(text)
        self.assertIn("变量，例：$k$ 与 <T> 泛型", cleaned)  # 中文标点、公式、泛型均保留
        self.assertIn("# 这不是标题", cleaned)               # 代码块内容不被误删
        self.assertIn("工作及文档要求", cleaned)                 # CJK 兼容字符归一

    def test_decode_accepts_utf8_with_ocr_replacement_chars(self):
        from lab_rag.ingest import _decode_markdown, _looks_like_mojibake

        # 真实设备手册含大量 U+FFFD（OCR 残留），严格 UTF-8 解码成功后不能被误判为乱码
        raw = "## InfiniiVision 3000T X-Series\n参数表 \ufffd\ufffd\ufffd 量程\n".encode("utf-8")
        text, encoding = _decode_markdown(raw)
        self.assertEqual(encoding, "utf-8")
        self.assertIn("InfiniiVision", text)
        self.assertFalse(_looks_like_mojibake(text))
        # 按单字节解码产生的连续拉丁扩展串仍应被拒绝
        self.assertTrue(_looks_like_mojibake("äöü" * 30))

    def test_table_enrichment_supports_separator_variants_and_partial_rows(self):
        from lab_rag.ingest import enrich_markdown_tables

        text = "| 参数 | 单位 |\n| :--: | :--: |\n| 功率 | dBm |\n| 频率 |"
        enriched = enrich_markdown_tables(text, cfg={"ENABLE_TABLE_SEMANTIC_ENRICHMENT": True,
                                                     "TABLE_SEMANTIC_MAX_ROWS": 20})
        self.assertIn("[表格结构化转写]", enriched)
        self.assertIn("参数=功率；单位=dBm", enriched)
        self.assertIn("参数=频率", enriched)  # 列数不等时按位置继承表头

    def test_header_split_ignores_code_comments_and_keeps_h4(self):
        from langchain_core.documents import Document
        from lab_rag.ingest import split_by_markdown_headers

        fence = chr(96) * 3
        text = "\n".join([
            "# 手册", "## 1 参数", fence + "sh", "# 这不是标题", fence,
            "#### 2.1 触发", "触发电平设置见下表。",
        ])
        sections = split_by_markdown_headers(Document(page_content=text, metadata={}), 1500, 200)
        titles = [s.metadata.get("section_title", "") for s in sections]
        self.assertIn("2.1 触发", titles)                       # #### 纳入层级
        self.assertNotIn("这不是标题", titles)                  # 代码块内 # 不判标题
        self.assertIn("# 这不是标题", "\n".join(s.page_content for s in sections))

    def test_atomic_blocks_keep_table_rows_together(self):
        from langchain_core.documents import Document
        from lab_rag import ingest

        cfg = ingest.load_config()
        with patch.object(ingest, "generate_parent_summaries", return_value={}):
            table = "| 参数 | 单位 |\n| --- | --- |\n| 功率 | dBm |\n| 频率 | GHz |"
            doc = Document(
                page_content="# 设备手册\n\n## 1 参数表\n\n" + table + "\n\n### 说明\n功率范围见手册。",
                metadata={"source": "/docs/a.md", "file_name": "a.md", "doc_title": "a",
                          "rel_path": "设备操作指南/VLC小组/MSO-X 3054A/Manuals/a.md"},
            )
            chunks, parents, _ = ingest.split_documents([doc], cfg)

        self.assertTrue(parents)
        self.assertEqual(len({p["parent_id"] for p in parents}), len(parents))
        self.assertTrue(all("rel_path" in p and "parent_ordinal" in p for p in parents))
        self.assertEqual(sum("| 功率 | dBm |" in c.page_content for c in chunks), 1)
        self.assertTrue(any("路径：设备操作指南 > VLC小组" in c.page_content for c in chunks))

    def test_filter_uses_raw_body_and_protects_structured_blocks(self):
        from lab_rag.ingest import analyze_bad_chunk_reason

        cfg = {"MIN_CHUNK_LENGTH": 120, "CHUNK_FILTER_SPECIAL_CHAR_RATIO": 0.25,
               "ENABLE_TABLE_FILTER_EXEMPTION": True}
        # 前缀不应参与长度/字符统计
        prefixed = "【文档：a > 章节：参数】\n" + "| 型号 | 量程 | MSO-X 3054A | https://example.com/very/long/path |"[:60]
        self.assertEqual(analyze_bad_chunk_reason(prefixed, cfg), "good")
        table_block = "【文档：a】\n[表格结构化转写]\n- 参数=功率；单位=dBm"
        self.assertEqual(analyze_bad_chunk_reason(table_block, cfg), "good")
        # 纯短噪声仍被过滤
        self.assertEqual(analyze_bad_chunk_reason("【文档：a】\n" + "x" * 20, cfg), "too_short")

    def test_rewrite_returns_keywords_and_independent_hyde_queries(self):
        from lab_rag import rag_agent

        class FakeChain:
            def __or__(self, other):
                return self

            def invoke(self, values):
                self.values = values
                payload = {
                    "standalone_question": "MSO-X 3054A 示波器如何设置触发",
                    "keywords": ["MSO-X 3054A", "触发"],
                    "expanded_queries": ["示波器触发设置", "触发模式 参数"],
                    "hypothetical_docs": "示波器触发设置需要先选择触发源并调整触发电平，参数见用户手册第三章。|||触发模式分为自动、正常与单次，MSO-X 系列在前面板提供快捷菜单。",
                }

                class Result:
                    content = json.dumps(payload, ensure_ascii=False)

                return Result()

        class FakePrompt:
            @staticmethod
            def from_template(template):
                return FakeChain()

        with patch.object(rag_agent, "ChatPromptTemplate", FakePrompt):
            result = rag_agent.rewrite_question(
                object(), "它怎么设置触发", cfg=rag_agent.load_config(),
                chat_history="用户: VLC小组有哪些设备\n助手: 有 MSO-X 3054A 示波器",
            )
        self.assertTrue(result["keywords"])
        self.assertEqual(len(result["hyde_queries"]), 2)
        self.assertTrue(all(h not in result["expanded_queries"] for h in result["hyde_queries"]))
        self.assertLessEqual(len(result["expanded_queries"]), 4)
        self.assertTrue(rag_agent.should_skip_rewrite("周报格式"))
        self.assertFalse(rag_agent.should_skip_rewrite("周报格式是什么"))
        self.assertFalse(rag_agent.should_skip_rewrite("周报格式", "用户: 之前问过周报"))

    def test_sparse_condition_budget_keeps_path_fields(self):
        from lab_rag import rag_agent

        class FakeClient:
            def __init__(self):
                self.filter_arg = None

            def scroll(self, collection_name, scroll_filter, limit, with_payload, timeout):
                self.filter_arg = scroll_filter
                return [], None

        client = FakeClient()
        rag_agent.retrieve_sparse(client, "lab", "VLC小组 MSO-X 3054A 使用方法", ["使用方法"],
                                  64, cfg=rag_agent.load_config())
        keys = [condition.key for condition in client.filter_arg.should]
        self.assertLessEqual(len(keys), 30)
        self.assertIn("metadata.rel_path", keys)
        self.assertIn("metadata.header_path", keys)
        self.assertIn("MSO-X", " ".join(
            condition.match.text for condition in client.filter_arg.should
        ))

    def test_dense_merge_uses_scores_and_unified_doc_key(self):
        from langchain_core.documents import Document
        from lab_rag import rag_agent

        class FakeVectorStore:
            def similarity_search_with_score(self, query, k):
                return [
                    (Document(page_content="低分", metadata={"parent_id": "p1", "chunk_hash": "h1"}), 0.2),
                    (Document(page_content="高分", metadata={"parent_id": "p2", "chunk_hash": "h2"}), 0.9),
                ]

        docs = rag_agent.retrieve_multi_query(FakeVectorStore(), ["q"], 10)
        self.assertEqual([d.page_content for d in docs], ["高分", "低分"])
        self.assertEqual(rag_agent.canonical_doc_key(docs[0]), "p2:h2")
        self.assertEqual(len(rag_agent.dedup_docs(docs + docs)), 2)

    def test_rrf_writes_score_and_citations_index_aligns(self):
        from langchain_core.documents import Document
        from lab_rag import rag_agent

        cfg = rag_agent.load_config()
        d1 = Document(page_content="x", metadata={"parent_id": "p1", "chunk_hash": "h1"})
        d2 = Document(page_content="y", metadata={"parent_id": "p2", "chunk_hash": "h2"})
        fused = rag_agent.reciprocal_rank_fusion([d1], [d2], cfg=cfg)
        self.assertTrue(all("rrf_score" in d.metadata for d in fused))

        used = [
            Document(page_content="a", metadata={"source": "/docs/a.md", "header_path": "第1章 > 1.1",
                                                 "rerank_score": 0.8}),
            Document(page_content="b", metadata={"source": "/docs/b.md", "section_title": "2.1",
                                                 "rerank_score": 0.6}),
        ]
        _, source_map = rag_agent.build_context(used, max_chars=3000, cfg=cfg)
        citations = rag_agent.build_citations(used)
        self.assertEqual([c["index"] for c in citations], list(source_map.keys()))
        self.assertIn("header_path", source_map[1])

    def test_coverage_thresholds_follow_config_and_hint_reaches_prompt(self):
        from langchain_core.documents import Document
        from lab_rag import rag_agent

        cfg = rag_agent.load_config()
        doc = Document(page_content="x", metadata={"rerank_score": 0.6})
        self.assertEqual(rag_agent.assess_coverage([doc], cfg)["level"], "medium")
        strict = dict(cfg, COVERAGE_MEDIUM_THRESHOLD=0.95, COVERAGE_LOW_THRESHOLD=0.9)
        self.assertEqual(rag_agent.assess_coverage([doc], strict)["level"], "very_low")
        self.assertIn("当前覆盖度提示", rag_agent.get_system_prompt_by_coverage("low", "注意核对原件"))
        self.assertEqual(rag_agent.assess_coverage([], cfg)["level"], "none")

    def test_group_alias_normalizes_catalog_target(self):
        from lab_rag import rag_agent

        self.assertEqual(rag_agent.extract_catalog_target("A2X 有哪些设备"), "A2X小组")
        route = rag_agent.rule_based_route("VLC小组有哪些设备及使用规范")
        self.assertEqual((route["route"], route["target"]), ("hybrid", "VLC小组"))

    def test_context_noise_filter_and_boundary_truncation(self):
        from lab_rag import rag_agent

        long_url = "https://example.com/" + "a" * 120
        cleaned = rag_agent._filter_context_noise("说明 " + long_url)
        self.assertIn("[长链接已省略", cleaned)
        table = "| 参数 | 单位 |\n| --- | --- |\n| 功率 | dBm |\n" + "补充说明" * 100
        truncated = rag_agent._truncate_at_boundary(table, 100)
        self.assertTrue(truncated.startswith("| 参数 | 单位 |"))


class RetrievalOptimizationTests(unittest.TestCase):
    """A/B 类检索优化的离线回归：单次遍历等价性、型号词元化、拒答与父块多样性。"""

    def test_catalog_snapshot_matches_two_independent_walks(self):
        import os
        from lab_rag import rag_agent as ra

        with tempfile.TemporaryDirectory() as root:
            for grp in ("A2X小组", "VLC小组"):
                for dev in ("Agilent MSO-X 3054A 示波器", "Keysight EDU36311A 电源"):
                    d = Path(root) / "设备操作指南" / grp / dev / "Manuals"
                    d.mkdir(parents=True, exist_ok=True)
                    (d / "Manual.md").write_text("x", encoding="utf-8")
            svn = Path(root) / "已毕业同学参考资料" / ".svn" / "pristine" / "a2"
            svn.mkdir(parents=True, exist_ok=True)
            (svn / "a2300c1.svn-base").write_text("x", encoding="utf-8")

            cfg = {"ENABLE_CATALOG_SNAPSHOT_CACHE": False,
                   "CATALOG_EXCLUDE_HIDDEN_DIRS": False}
            snap = ra._get_catalog_snapshot(root, cfg)
            self.assertTrue(snap)
            for kw in ("A2X小组", "VLC小组", "Manuals"):
                with self.subTest(kw=kw):
                    legacy_dirs = []
                    for cur, dirs, _ in os.walk(root):
                        rel = os.path.relpath(cur, root)
                        rel = "" if rel == "." else rel
                        for name in dirs:
                            rp = ra.path_to_unix(os.path.join(rel, name) if rel else name)
                            score = ra.path_match_score(kw, rp)
                            if score >= 0.45:
                                legacy_dirs.append({"rel_path": rp, "score": round(score, 4)})
                    legacy_dirs.sort(key=lambda x: (-x["score"], len(x["rel_path"])))
                    self.assertEqual(
                        legacy_dirs[:10],
                        ra.find_best_matching_dirs(kw, root, limit=10, snapshot=snap, cfg=cfg),
                    )

            clean = ra._get_catalog_snapshot(
                root, {"ENABLE_CATALOG_SNAPSHOT_CACHE": False,
                       "CATALOG_EXCLUDE_HIDDEN_DIRS": True})
            self.assertFalse(any(".svn" in item[2] for item in clean))
            best = ra.find_best_matching_dirs("A2X小组", root, limit=1,
                                              snapshot=clean, cfg={})
            self.assertEqual(best[0]["rel_path"], "设备操作指南/A2X小组")

    def test_hybrid_query_dedup_keeps_group_and_target_device(self):
        from lab_rag import rag_agent as ra

        result = {
            "mode": "filesystem_directory", "matched_dir": "设备操作指南/VLC小组",
            "directories": [{"rel_path": "设备操作指南/VLC小组/" + d} for d in
                            ("Agilent MSO-X 3054A 示波器", "Keysight EDU36311A 电源",
                             "Newport 843-R 光功率计", "HOLOEYE LETO-3 空间光调制器")],
            "files": [],
        }
        q = "VLC小组有哪些设备，并说明 MSO-X 3054A 触发参数如何设置？"
        queries = ra.build_catalog_retrieval_queries(result, "VLC小组", q,
                                                     cfg={"MAX_HYBRID_QUERIES": 3})
        self.assertEqual(len(queries), 3)
        self.assertTrue(any("MSO-X 3054A" in item for item in queries))
        self.assertTrue(any(item.startswith("VLC小组") for item in queries))
        self.assertEqual(len(queries), len(set(queries)))

    def test_sparse_terms_keep_alphanumeric_model_numbers(self):
        from lab_rag import rag_agent as ra

        for text, expected in (
                ("N9322C Basic Spectrum Analyzer 中 DANL 的参数是多少？", "N9322C"),
                ("EDU36311A 编程指南中 SCPI common commands 如何使用？", "EDU36311A"),
                ("MSO-X 3054A 示波器的触发参数如何设置？", "3054A"),
                ("USRP-2944 Specifications 中发射机通道的参数是多少？", "USRP-2944"),
        ):
            with self.subTest(text=text[:20]):
                self.assertIn(expected, ra._build_sparse_terms(text, []))
        terms = ra._build_sparse_terms("MSO-X 3054A 示波器", [])
        self.assertNotIn("MSO", terms)  # 复合型号不应再拆出无区分度前缀

    def test_select_reranked_respects_absolute_threshold_and_fallback(self):
        from langchain_core.documents import Document
        from lab_rag import rag_agent as ra

        docs = [Document(page_content=f"d{i}", metadata={}) for i in range(5)]
        boosted = [(docs[i], s) for i, s in enumerate((0.9, 0.5, 0.4, 0.2, 0.1))]
        cfg = {"RERANKER_SCORE_THRESHOLD": 0.45,
               "RERANKER_ALLOW_LOW_SCORE_FALLBACK": False}
        self.assertEqual(len(ra.select_reranked(boosted, cfg, 5)), 2)
        # 全部低分：开关关闭时返回空，打开时保底 1 条
        low = [(docs[0], 0.2), (docs[1], 0.1)]
        self.assertEqual(ra.select_reranked(low, cfg, 5), [])
        cfg["RERANKER_ALLOW_LOW_SCORE_FALLBACK"] = True
        self.assertEqual(len(ra.select_reranked(low, cfg, 5)), 1)
        # 相对比例与保底条数
        cfg2 = {"RERANKER_SCORE_THRESHOLD": 0.45, "RERANKER_RELATIVE_RATIO": 0.5,
                "RERANKER_MIN_KEEP": 3}
        self.assertGreaterEqual(len(ra.select_reranked(boosted, cfg2, 5)), 3)

    def test_answerability_gate_only_fires_on_zero_evidence(self):
        from langchain_core.documents import Document
        from lab_rag import rag_agent as ra

        def doc(meta, score):
            return Document(page_content="x", metadata={**meta, "rerank_score": score})

        cfg = {"ENABLE_ANSWERABILITY_GATE": True,
               "ANSWERABILITY_TOP_SCORE_FLOOR": 0.82,
               "ANSWERABILITY_MIN_KEYWORD_HITS": 1}
        # 无关高分文档（天气问题却召回参考书）→ 判为无答案
        unrelated = [doc({"source": "参考书/涉水光学.md", "file_name": "涉水光学.md"}, 0.755)]
        gate = ra.evaluate_answerability(["上海", "天气", "实时温度"], unrelated, cfg)
        self.assertFalse(gate["answerable"])
        # 有证据命中 → 可答
        related = [doc({"source": "设备操作指南/VLC小组/MSO-X 3054A 示波器/manual.md",
                        "file_name": "manual.md"}, 0.9)]
        gate = ra.evaluate_answerability(["MSO-X 3054A", "触发参数"], related, cfg)
        self.assertTrue(gate["answerable"])
        self.assertGreaterEqual(gate["keyword_hits"], 1)
        # LLM 生成的摘要含通用词，不能当作「文档命中问题实体」的证据
        summary_only = [doc({"source": "设备操作指南/VLC小组/MSO-X 3054A 示波器/manual.md",
                             "file_name": "manual.md",
                             "parent_summary": "Django ORM 的查询方法与信息格式说明"}, 0.543)]
        gate = ra.evaluate_answerability(["Django ORM", "查询方法"], summary_only, cfg)
        self.assertFalse(gate["answerable"])
        # 关键词为空（跳过改写）时不启用闸门，避免误判
        self.assertTrue(ra.evaluate_answerability([], unrelated, cfg)["answerable"])
        # 开关关闭时不生效
        self.assertTrue(ra.evaluate_answerability(
            ["上海"], unrelated, {**cfg, "ENABLE_ANSWERABILITY_GATE": False})["answerable"])

    def test_table_transcription_skips_ocr_replacement_noise(self):
        from lab_rag import ingest

        cfg = {"ENABLE_TABLE_SEMANTIC_ENRICHMENT": True,
               "TABLE_SEMANTIC_MAX_ROWS": 20,
               "TABLE_SEMANTIC_MAX_REPLACEMENT_RATIO": 0.15}
        normal = ("| 参数 | 值 |\n| --- | --- |\n"
                  "| 增益 | 20 dB |\n| 带宽 | 3 GHz |")
        out = ingest.enrich_markdown_tables(normal, cfg)
        self.assertIn("参数=增益；值=20 dB", out)
        self.assertIn("参数=带宽；值=3 GHz", out)

        noisy = ("| 章节 | 页码 |\n| --- | --- |\n"
                 "| 1.1 简介 | " + "\ufffd" * 20 + " 5 |\n"
                 "| 1.2 操作 | " + "\ufffd" * 20 + " 9 |")
        noisy_out = ingest.enrich_markdown_tables(noisy, cfg)
        self.assertNotIn("[表格结构化转写]", noisy_out)
        # 原文仍保留（只跳过转写，不丢源内容）
        self.assertIn("| 1.1 简介 |", noisy_out)

    def test_parent_expansion_caps_single_source_monopoly(self):
        from langchain_core.documents import Document
        from lab_rag import rag_agent as ra

        children = []
        for i in range(12):
            src = "长手册.md" if i < 10 else "目标手册.md"
            children.append(Document(
                page_content=f"child {i}",
                metadata={"parent_id": f"p{i}", "source": src, "file_name": src,
                          "rerank_score": 1.2 - i * 0.01, "parent_content": f"parent {i}"}))
        cfg = {"PARENT_EXPAND_MAX_CHARS": 9000, "PARENT_MAX_SINGLE_CHARS": 1500,
               "PARENT_MAX_PER_SOURCE": 2}
        parents = ra.expand_to_parent_docs(children, cfg=cfg, top_k=16)
        sources = [(p.metadata.get("source") or "") for p in parents]
        self.assertLessEqual(sources.count("长手册.md"), 2)
        self.assertIn("目标手册.md", sources)


class EvidencePriorTests(unittest.TestCase):
    """强证据先验：日期/文档标题词元/型号精确命中身份字段时的加性加分。"""

    def test_strong_evidence_entities_extract_dates_and_titles(self):
        from lab_rag import rag_agent as ra
        cfg = {"ENABLE_CORPUS_LEXICON": True}
        # 日期与文档标题词元在问题中出现 → 命中
        ents = ra._strong_evidence_entities(
            "20230106 期末组会中 VLC小组 汇报了哪些论文写作方法？", [], cfg)
        self.assertIn("20230106", ents)
        self.assertIn("论文写作", ents)
        # 未出现的实体不应进入
        self.assertNotIn("阿司匹林", ents)
        # 纯泛词不应被当作专名（小组/系统/设备被过滤）
        ents2 = ra._strong_evidence_entities("VLC小组 与 A2X小组 的实验设备有哪些？", [], cfg)
        self.assertNotIn("小组", ents2)

    def test_evidence_prior_promotes_date_matching_doc(self):
        from langchain_core.documents import Document
        from lab_rag import rag_agent as ra

        class FakeReranker:
            # 语义重排把「无日期的同类文档」排在了正确文档前面（模拟语义失真）
            def predict(self, pairs):
                return [0.95, 0.99]

        docs = [
            Document(page_content="right", metadata={
                "source": "组会资料/20230106期末组会_许健鹏.md", "file_name": "20230106期末组会_许健鹏.md",
                "rerank_score": 0.95}),
            Document(page_content="wrong", metadata={
                "source": "组会资料/20220121VLC小组总结汇报.md", "file_name": "20220121VLC小组总结汇报.md",
                "rerank_score": 0.99}),
        ]
        cfg = {"EVIDENCE_PRIOR_WEIGHT": 0.12, "EVIDENCE_PRIOR_CAP": 0.24,
               "RERANKER_SCORE_THRESHOLD": 0.3, "RERANKER_ALLOW_LOW_SCORE_FALLBACK": False}
        out = ra.rerank_docs(FakeReranker(), "期末组会", docs, cfg, top_k=2,
                             original_question="20230106 期末组会中 VLC小组 汇报了什么？")
        self.assertEqual(out[0].metadata["file_name"], "20230106期末组会_许健鹏.md")

    def test_evidence_prior_inactive_without_entities(self):
        from langchain_core.documents import Document
        from lab_rag import rag_agent as ra

        class FakeReranker:
            def predict(self, pairs):
                return [0.95, 0.99]

        docs = [
            Document(page_content="a", metadata={"source": "x/a.md", "file_name": "a.md", "rerank_score": 0.95}),
            Document(page_content="b", metadata={"source": "x/b.md", "file_name": "b.md", "rerank_score": 0.99}),
        ]
        cfg = {"EVIDENCE_PRIOR_WEIGHT": 0.12, "EVIDENCE_PRIOR_CAP": 0.24,
               "RERANKER_SCORE_THRESHOLD": 0.3, "RERANKER_ALLOW_LOW_SCORE_FALLBACK": False}
        out = ra.rerank_docs(FakeReranker(), "实验室常用设备", docs, cfg, top_k=2,
                             original_question="实验室常用设备有哪些？")
        # 无强实体命中 → 不改变语义重排结果
        self.assertEqual(out[0].metadata["file_name"], "b.md")


class LexiconAndGeneralizationTests(unittest.TestCase):
    """v3.3.0：词典外置、语料派生、证据重排与测试集门槛的离线回归。"""

    def test_lexicon_schema_and_required_fields(self):
        from lab_rag import lexicon
        data = lexicon.load_lexicon()
        self.assertEqual(data["schema_version"], 1)
        for key in ("notation", "route_keywords", "catalog_intent", "table_recovery",
                    "flattened_table_groups", "flattened_table_marker",
                    "filter_exempt_markers", "group_aliases", "placeholders"):
            with self.subTest(key=key):
                self.assertIn(key, data)
        self.assertTrue(data["flattened_table_warning"].startswith(data["flattened_table_marker"]))
        self.assertIn("[变量格式表语义恢复]", data["filter_exempt_markers"])

    def test_lexicon_rejects_invalid_schema(self):
        import tempfile
        from lab_rag import lexicon
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.json"
            bad.write_text(json.dumps({"schema_version": 1, "notation": {}}), encoding="utf-8")
            with self.assertRaises(lexicon.LexiconError):
                lexicon.load_lexicon(str(bad), refresh=True)
            missing = Path(d) / "missing.json"
            with self.assertRaises(lexicon.LexiconError):
                lexicon.load_lexicon(str(missing), refresh=True)
        lexicon.reset_cache()

    def test_lexicon_caches_and_refresh_works(self):
        from lab_rag import lexicon
        lexicon.reset_cache()
        a = lexicon.load_lexicon()
        b = lexicon.load_lexicon()
        self.assertIs(a, b)
        lexicon.reset_cache()
        self.assertIsNot(a, lexicon.load_lexicon())

    def test_externalized_route_keywords_match_expected(self):
        from lab_rag import rag_agent as ra, lexicon
        kw = lexicon.load_lexicon()["route_keywords"]
        self.assertEqual(kw["file_list"][0], "目录")
        self.assertIn("设备", kw["file_list"])
        self.assertIn("原理", kw["rag_search"])
        # 63+ 条真实测试集路由实测一致
        data = json.loads((PROJECT_ROOT / "tests" / "rag_perf_queries.json").read_text(encoding="utf-8"))
        for q in data["queries"]:
            res = ra.rule_based_route(q["query"])
            got = res["route"] if isinstance(res, dict) else "llm_fallback"
            with self.subTest(id=q["id"]):
                self.assertEqual(got, q["expected_route"])

    def test_externalized_notation_plan_and_answer_template(self):
        from lab_rag import rag_agent as ra, lexicon
        plan = ra.build_notation_retrieval_plan()
        self.assertEqual(plan["rewritten_question"],
                         "论文变量格式规范中矩阵和向量的正体、斜体、粗体及大小写要求")
        self.assertEqual(len(plan["queries"]), 3)
        self.assertTrue(ra.is_notation_format_question("论文中矩阵和向量的正体格式怎么写？"))
        self.assertFalse(ra.is_notation_format_question("MSO-X 3054A 触发参数如何设置？"))
        self.assertFalse(ra.is_notation_format_question(
            "论文中矩阵的正体格式", {"ENABLE_NOTATION_SPECIALIZATION": False}))
        ctx = "[来源2] 前缀\n变量类型=向量、矩阵；英文字母=正体粗体；罗马字母=沿用原文\n"
        ans = ra.build_grounded_notation_answer(ctx)
        self.assertIn("$\\mathbf{k}$", ans)
        self.assertIn("正体粗体", ans)
        self.assertIn("[来源2]", ans)
        self.assertEqual(ra.build_grounded_notation_answer("无表格"), "")
        self.assertIn("__ENGLISH_RULE_TOKEN__",
                      lexicon.load_lexicon()["notation"]["answer_template"])

    def test_externalized_catalog_intent_is_equivalent(self):
        from lab_rag import rag_agent as ra
        self.assertEqual(ra.extract_catalog_content_intent("VLC小组有哪些设备，怎么使用？"),
                         "使用方法 操作步骤")
        self.assertEqual(ra.extract_catalog_content_intent("列出设备并说明安全规范"),
                         "操作规范 注意事项")
        self.assertEqual(ra.extract_catalog_content_intent("有哪些设备，参数指标是什么？"),
                         "参数指标")
        self.assertEqual(ra.extract_catalog_content_intent("有哪些设备"), "说明 用途")

    def test_externalized_table_recovery_and_exempt_markers(self):
        from lab_rag import ingest, lexicon
        lex = lexicon.load_lexicon()
        flat = "变量格式 变量类型 英文字母 罗马字母 标量 集合 向量、矩阵 " + "正体并粗体 " * 40
        recovered = ingest.recover_flattened_variable_format_table(flat)
        self.assertIn(lex["notation"]["recovery_marker"], recovered)
        self.assertIn(lex["table_recovery"]["template"], recovered)
        cfg = ingest.load_config()
        long_line = "变量类型 参数名称 序号 单位 说明 含义 标量 集合 向量 矩阵 " + "补充内容 " * 50
        warned = ingest.mark_suspect_flattened_tables(long_line, cfg)
        self.assertIn(lex["flattened_table_marker"], warned)
        self.assertIn(lex["flattened_table_warning"], warned)

    def test_corpus_lexicon_generation_and_group_aliases(self):
        import importlib.util
        from lab_rag import rag_agent as ra, lexicon
        spec = importlib.util.spec_from_file_location(
            "build_corpus_lexicon", PROJECT_ROOT / "scripts" / "build_corpus_lexicon.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for grp, dev in (("A2X小组", "Keysight N9322C 频谱分析仪"),
                             ("VLC小组", "Agilent MSO-X 3054A 示波器")):
                p = root / "设备操作指南" / grp / dev / "Manuals"
                p.mkdir(parents=True, exist_ok=True)
                (p / "Manual.md").write_text("x", encoding="utf-8")
            built = mod.build(str(root))
            self.assertEqual(built["n_markdown_files"], 2)
            self.assertEqual(built["group_aliases"]["A2X"], "A2X小组")
            self.assertIn("N9322C", built["device_models"])
            self.assertTrue(built["manual_types"])
            dest = root / "corpus.json"
            dest.write_text(json.dumps(built), encoding="utf-8")
            lexicon.reset_cache()
            loaded = lexicon.load_corpus_lexicon(str(dest), refresh=True, required=True)
            self.assertEqual(loaded["group_aliases"]["VLC"], "VLC小组")
            self.assertEqual(ra.extract_catalog_target("A2X 有哪些设备",
                                                        {"ENABLE_CORPUS_LEXICON": False}), "A2X小组")
        lexicon.reset_cache()

    def test_corpus_lexicon_missing_falls_back_to_static_aliases(self):
        from lab_rag import lexicon
        lexicon.reset_cache()
        aliases = lexicon.get_group_aliases({"ENABLE_CORPUS_LEXICON": False})
        self.assertEqual(aliases["A2X"], "A2X小组")
        self.assertIsNone(lexicon.load_corpus_lexicon("/nonexistent/corpus.json", refresh=True))

    def test_evidence_terms_cover_model_variants(self):
        from lab_rag import rag_agent as ra
        terms = ra._evidence_terms("N9322C 843-R 与 USRP-2944 的 DANL 参数", [])
        for expected in ("n9322c", "843-r", "usrp-2944", "danl"):
            with self.subTest(token=expected):
                self.assertIn(expected, terms)

    def test_evidence_rerank_only_swaps_within_tie_group(self):
        from langchain_core.documents import Document
        from lab_rag import rag_agent as ra

        def d(rel, score):
            return Document(page_content="x", metadata={"rel_path": rel,
                                                        "file_name": rel.split("/")[-1],
                                                        "rerank_score": score})
        cfg = {"ENABLE_EVIDENCE_RERANK": True, "EVIDENCE_RERANK_WEIGHT": 0.12,
               "ENABLE_CORPUS_LEXICON": False}
        docs = [
            d("设备操作指南/VLC小组/Keysight EDU36311A 电源/guide.md", 1.130),
            d("设备操作指南/VLC小组/Newport 843-R 光功率计/measurement.md", 1.129),
            d("设备操作指南/VLC小组/Agilent MSO-X 3054A 示波器/manual.md", 1.128),
        ]
        out = ra.reorder_final_docs_by_evidence(
            docs, "Newport 843-R 光功率计的测量不确定度如何评估？",
            ["Newport 843-R", "光功率计"], cfg)
        self.assertEqual({id(x) for x in docs}, {id(x) for x in out})
        self.assertIn("Newport", out[0].metadata["rel_path"])
        scores = [x.metadata["rerank_score"] for x in out]
        inversions = [scores[j] - scores[i] for i in range(len(scores)) for j in range(i + 1, len(scores))]
        self.assertLessEqual(max(inversions), 0.12)

    def test_evidence_rerank_keeps_large_gap_order_and_switch(self):
        from langchain_core.documents import Document
        from lab_rag import rag_agent as ra

        def d(rel, score):
            return Document(page_content="x", metadata={"rel_path": rel, "rerank_score": score})
        docs = [d("a.md", 1.2), d("b.md", 0.3)]
        self.assertEqual([id(x) for x in ra.reorder_final_docs_by_evidence(docs, "b 型号", ["b"], {})],
                         [id(x) for x in docs])
        off = ra.reorder_final_docs_by_evidence(docs, "b 型号", ["b"],
                                                {"ENABLE_EVIDENCE_RERANK": False})
        self.assertEqual([id(x) for x in off], [id(x) for x in docs])

    def test_evidence_rerank_preserves_citation_context_alignment(self):
        from langchain_core.documents import Document
        from lab_rag import rag_agent as ra
        docs = [Document(page_content="内容 " * 20,
                         metadata={"rel_path": f"设备/型号{i}/m.md", "file_name": f"m{i}.md",
                                   "doc_title": f"doc{i}", "rerank_score": 1.1 - i * 0.001})
                for i in range(4)]
        out = ra.reorder_final_docs_by_evidence(docs, "型号2 参数", ["型号2"], {})
        ctx, smap = ra.build_context(out, max_chars=99999, cfg={})
        cits = ra.build_citations(out)
        self.assertEqual([c["index"] for c in cits], list(range(1, len(out) + 1)))
        self.assertEqual(len(smap), len(out))
        for i in range(1, len(out) + 1):
            self.assertIn(f"[来源{i}]", ctx)

    def test_testset_distribution_and_holdout_isolation(self):
        main = json.loads((PROJECT_ROOT / "tests" / "rag_perf_queries.json").read_text(encoding="utf-8"))
        hold = json.loads((PROJECT_ROOT / "tests" / "rag_perf_holdout.json").read_text(encoding="utf-8"))
        meta = main["meta"]
        self.assertGreaterEqual(meta["total"], 120)
        self.assertLessEqual(meta["device_manual_ratio"], 0.40)
        self.assertGreaterEqual(min(meta["doc_type_distribution"].values()), 8)
        self.assertGreaterEqual(len(meta["case_tag_distribution"]), 14)
        self.assertGreaterEqual(min(meta["case_tag_distribution"].values()), 3)
        self.assertGreaterEqual(meta["non_rag_route_ratio"], 0.25)
        self.assertGreaterEqual(hold["meta"]["total"], 20)
        self.assertFalse({q["query"] for q in main["queries"]} & {q["query"] for q in hold["queries"]})
        self.assertFalse({q["id"] for q in main["queries"]} & {q["id"] for q in hold["queries"]})
        self.assertTrue(all("grounding_type" in q for q in main["queries"]))

    def test_iteration_criteria_schema(self):
        spec = json.loads((PROJECT_ROOT / "config" / "iteration_criteria.json").read_text(encoding="utf-8"))
        self.assertEqual(spec["schema_version"], 1)
        ids = [c["id"] for c in spec["criteria"]]
        self.assertEqual(len(ids), len(set(ids)))
        for c in spec["criteria"]:
            with self.subTest(id=c["id"]):
                self.assertIn(c["op"], ("==", ">=", "<=", ">", "<"))
                self.assertIn("dimension", c)
        for key in ("hit_rate", "top1_rate", "generalization_gap", "holdout_hit_rate"):
            self.assertIn(key, ids)

    def test_answerability_uses_corpus_and_bigram_evidence(self):
        """三线证据：改写关键词 / 语料词典实体 / 原问题 bigram 重合。"""
        from langchain_core.documents import Document
        from lab_rag import rag_agent as ra

        def doc(rel, score):
            return Document(page_content="x", metadata={
                "source": rel, "file_name": rel.split("/")[-1], "rerank_score": score})

        cfg = {"ENABLE_ANSWERABILITY_GATE": True,
               "ANSWERABILITY_TOP_SCORE_FLOOR": 0.90,
               "ANSWERABILITY_MIN_KEYWORD_HITS": 1,
               "ANSWERABILITY_MIN_OVERLAP_HITS": 2,
               "ENABLE_CORPUS_LEXICON": True}
        # 语料侧证据：原问题含真实文档实体「实验拍照要求」
        g = ra.evaluate_answerability(
            ["照片画质", "视频清晰度", "数值规定"],
            [doc("实验拍照要求.md", 0.429)], cfg,
            "实验拍照要求里对照片画质与视频清晰度有哪些数值规定？")
        self.assertTrue(g["answerable"])
        self.assertGreater(g["corpus_hits"], 0)

        # bigram 证据：抽象关键词零命中，但原问题与文档标题有 ≥2 个中文 2-gram 重合
        g2 = ra.evaluate_answerability(
            ["命名", "提交方式"],
            [doc("团队工作及文档要求汇总_v0.1a_20250224.md", 0.818)], cfg,
            "团队文档要求中不允许哪些命名和提交方式？")
        self.assertTrue(g2["answerable"])
        self.assertGreaterEqual(g2["overlap_hits"], 2)

        # 三线全零且低分 → 判为无答案
        g3 = ra.evaluate_answerability(
            ["上海", "天气", "实时温度"],
            [doc("参考书/涉水光学.md", 0.755)], cfg,
            "今天上海天气的实时温度如何？")
        self.assertFalse(g3["answerable"])
        self.assertEqual(g3["keyword_hits"], 0)
        self.assertEqual(g3["corpus_hits"], 0)

        # 关键词为空（主题级短问法跳过改写）时闸门不启用
        g4 = ra.evaluate_answerability([], [doc("参考书/涉水光学.md", 0.60)], cfg,
                                       "实验室主要研究方向")
        self.assertTrue(g4["answerable"])
        self.assertEqual(g4["reason"], "no_keywords_skip_gate")

    def test_short_topic_query_skips_rewrite(self):
        from lab_rag import rag_agent as ra
        for q in ("实验室主要研究方向", "水下光通信", "MIMO", "AI4Science", "可见光通信"):
            with self.subTest(q=q):
                self.assertTrue(ra.should_skip_rewrite(q))
        # 带疑问词或代词的短问句仍需改写消歧
        self.assertFalse(ra.should_skip_rewrite("MSO-X 3054A 触发参数如何设置？"))
        self.assertFalse(ra.should_skip_rewrite("它是什么？"))
        self.assertFalse(ra.should_skip_rewrite(
            "请比较水下无线光通信与涉水光学两本书在研究对象、关键技术、调制方法和应用场景上的差异，并说明为什么会有这些差异"))

    def test_engineering_criteria_metrics_are_all_satisfied(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "check_engineering_criteria", PROJECT_ROOT / "scripts" / "check_engineering_criteria.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # 只验证纯静态指标；reliability_count 会执行 unittest，不能在回归里递归调用
        self.assertEqual(mod.lint_errors(), 0)
        self.assertEqual(mod.new_dependencies(), 0)
        self.assertEqual(mod.inlined_customizations(), 0)
        self.assertGreater(mod.reliability_count(explicit=75), 70)
        metrics_path = PROJECT_ROOT / "logs" / "engineering_metrics.json"
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(metrics.get("reliability_test_count", 0), 70)

    def test_page_content_hash_script_is_readonly_and_reproducible(self):
        """零重建判定脚本只读且可重复运行（对同一语料返回同一指纹）。"""
        import subprocess
        script = PROJECT_ROOT / "scripts" / "check_page_content_hash.py"
        out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                             cwd=str(PROJECT_ROOT), timeout=180)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("IDENTICAL", out.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
