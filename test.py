"""
env_check.py

功能说明：
1. 读取 .env 配置文件
2. 检查关键 Python 依赖是否可导入
3. 检查本地 vLLM OpenAI 兼容接口是否可访问
4. 检查远程 Qdrant 是否可连接（如果尚未部署，会提示错误）

运行方式：
    python env_check.py
"""

import os
import sys
from typing import Optional

import requests
from dotenv import load_dotenv


def print_title(title: str) -> None:
    """打印清晰的分隔标题，方便查看控制台输出。"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_config() -> dict:
    """
    加载 .env 配置文件，并返回配置字典。

    Returns:
        dict: 包含项目所需核心配置项的字典
    """
    try:
        load_dotenv()

        config = {
            "VLLM_BASE_URL": os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
            "VLLM_API_KEY": os.getenv("VLLM_API_KEY", ""),
            "VLLM_MODEL_NAME": os.getenv("VLLM_MODEL_NAME", ""),
            "QDRANT_HOST": os.getenv("QDRANT_HOST", "127.0.0.1"),
            "QDRANT_PORT": int(os.getenv("QDRANT_PORT", "6333")),
            "QDRANT_COLLECTION": os.getenv("QDRANT_COLLECTION", "lab_knowledge_base"),
            "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY", ""),
            "EMBEDDING_MODEL_NAME": os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3"),
            "EMBEDDING_DEVICE": os.getenv("EMBEDDING_DEVICE", "cuda"),
            "DOCS_PATH": os.getenv("DOCS_PATH", "./data"),
            "CHUNK_SIZE": int(os.getenv("CHUNK_SIZE", "800")),
            "CHUNK_OVERLAP": int(os.getenv("CHUNK_OVERLAP", "150")),
        }

        return config

    except Exception as e:
        print(f"[ERROR] 加载 .env 配置失败: {e}")
        raise


def check_python_env() -> None:
    """打印当前 Python 运行环境信息。"""
    print_title("1. Python 运行环境检查")
    try:
        print(f"Python 可执行文件: {sys.executable}")
        print(f"Python 版本: {sys.version}")
    except Exception as e:
        print(f"[ERROR] Python 环境检查失败: {e}")


def check_package(package_name: str) -> None:
    """
    检查单个包是否可导入，并打印版本号。

    Args:
        package_name (str): Python 包名
    """
    try:
        module = __import__(package_name)
        version = getattr(module, "__version__", "未知版本")
        print(f"[OK] {package_name:<25} 版本: {version}")
    except Exception as e:
        print(f"[WARN] {package_name:<25} 导入失败: {e}")


def check_dependencies() -> None:
    """检查后续 RAG 项目所需关键依赖。"""
    print_title("2. 关键依赖检查")
    packages = [
        "langchain",
        "langchain_openai",
        "langchain_community",
        "qdrant_client",
        "sentence_transformers",
        "streamlit",
        "docx2txt",
        "fitz",
        "dotenv",
    ]
    for pkg in packages:
        check_package(pkg)


def check_vllm_api(base_url: str, api_key: str) -> None:
    """
    检查本地 vLLM OpenAI 兼容接口是否可用。

    Args:
        base_url (str): vLLM API 基础地址
        api_key (str): API Key
    """
    print_title("3. vLLM 接口连通性检查")

    url = f"{base_url}/models"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        print(f"请求地址: {url}")
        print(f"HTTP 状态码: {response.status_code}")

        response.raise_for_status()
        data = response.json()

        print("[OK] vLLM 接口访问成功。")
        print("返回结果：")
        print(data)

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] vLLM 接口请求失败: {e}")
    except Exception as e:
        print(f"[ERROR] vLLM 接口检查异常: {e}")


def check_qdrant(host: str, port: int, api_key: Optional[str] = None) -> None:
    """
    检查远程 Qdrant REST 接口是否可访问。

    Args:
        host (str): Qdrant 服务器地址
        port (int): Qdrant REST 端口
        api_key (Optional[str]): Qdrant API Key
    """
    print_title("4. Qdrant 接口连通性检查")

    url = f"http://{host}:{port}/collections"
    headers = {}

    if api_key:
        headers["api-key"] = api_key

    try:
        response = requests.get(url, headers=headers, timeout=30)
        print(f"请求地址: {url}")
        print(f"HTTP 状态码: {response.status_code}")

        response.raise_for_status()
        data = response.json()

        print("[OK] Qdrant 接口访问成功。")
        print("返回结果：")
        print(data)

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Qdrant 接口请求失败: {e}")
    except Exception as e:
        print(f"[ERROR] Qdrant 接口检查异常: {e}")


def print_config(config: dict) -> None:
    """打印当前关键配置，便于人工核对。"""
    print_title("5. 当前项目配置")
    for k, v in config.items():
        print(f"{k}: {v}")


def main() -> None:
    """主函数：按顺序执行环境检查。"""
    try:
        config = load_config()

        check_python_env()
        check_dependencies()
        print_config(config)

        check_vllm_api(
            base_url=config["VLLM_BASE_URL"],
            api_key=config["VLLM_API_KEY"]
        )

        check_qdrant(
            host=config["QDRANT_HOST"],
            port=config["QDRANT_PORT"],
            api_key=config["QDRANT_API_KEY"] or None
        )

        print_title("检查完成")
        print("如果 vLLM 检查成功，说明本地模型服务环境已经就绪。")
        print("如果 Qdrant 检查失败，请先等待 CPU 服务器完成 Qdrant 部署。")

    except Exception as e:
        print(f"[FATAL] 环境检测脚本执行失败: {e}")


if __name__ == "__main__":
    main()