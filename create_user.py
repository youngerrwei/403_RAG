"""安全地创建本地用户或原子更新其密码。"""

import getpass
import hashlib
import json
import os
import secrets
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from logger import get_logger


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")
_configured_users_file = Path(os.getenv("USERS_FILE", "config/users.json"))
USERS_FILE = (
    _configured_users_file
    if _configured_users_file.is_absolute()
    else PROJECT_ROOT / _configured_users_file
)
PBKDF2_ITERATIONS = 200000
_logger = get_logger("create_user")


def log(message: str) -> None:
    _logger.info(message)


def load_users() -> list:
    """读取用户文件；格式损坏时必须终止，禁止把现有数据静默覆盖为空。"""
    if not USERS_FILE.exists():
        log(f"用户文件不存在，将在保存时创建: {USERS_FILE}")
        return []

    try:
        with USERS_FILE.open("r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _logger.error("加载用户文件失败: %s", exc)
        raise ValueError(f"无法安全读取用户文件: {USERS_FILE}") from exc

    if not isinstance(data, list):
        raise ValueError(f"用户文件顶层必须是 JSON 数组: {USERS_FILE}")
    return data


def save_users(users: list) -> None:
    """在同目录写临时文件并原子替换，避免中途失败损坏凭据文件。"""
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=USERS_FILE.parent,
            prefix=f".{USERS_FILE.name}.",
            suffix=".tmp",
            delete=False,
        ) as file_obj:
            temp_path = Path(file_obj.name)
            json.dump(users, file_obj, ensure_ascii=False, indent=2)
            file_obj.write("\n")
            file_obj.flush()
            os.fsync(file_obj.fileno())

        os.chmod(temp_path, 0o600)
        os.replace(temp_path, USERS_FILE)
        log(f"用户文件保存成功: {USERS_FILE}")
    except Exception:  # JSON 序列化和文件系统失败都必须清理临时文件。
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        _logger.exception("保存用户文件失败: %s", USERS_FILE)
        raise


def make_password_record(password: str) -> dict:
    """使用 PBKDF2-HMAC-SHA256 生成密码记录。"""
    salt = secrets.token_bytes(16)
    derived_key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return {
        "salt": salt.hex(),
        "password_hash": derived_key.hex(),
        "hash_method": "pbkdf2_sha256",
        "iterations": PBKDF2_ITERATIONS,
    }


def find_user(users, username: str):
    for user in users:
        if isinstance(user, dict) and user.get("username") == username:
            return user
    return None


def main() -> int:
    log("本地用户创建工具（PBKDF2 版本）")
    try:
        username = input("请输入用户名: ").strip()
        if not username:
            _logger.error("用户名不能为空")
            return 1

        password = getpass.getpass("请输入密码: ").strip()
        if not password:
            _logger.error("密码不能为空")
            return 1

        confirm_password = getpass.getpass("请再次输入密码确认: ").strip()
        if password != confirm_password:
            _logger.error("两次输入的密码不一致")
            return 1

        users = load_users()
        password_record = make_password_record(password)

        existing_user = find_user(users, username)

        if existing_user:
            existing_user.update(password_record)
            log(f"用户已存在，密码已更新: {username}")
        else:
            users.append({"username": username, **password_record})
            log(f"新用户创建成功: {username}")

        save_users(users)
        # 不输出盐值和密码哈希，避免凭据材料进入终端历史或采集日志。
        log(f"操作完成；当前用户数: {len(users)}")
        return 0

    except KeyboardInterrupt:
        _logger.warning("用户取消操作")
        return 130
    except Exception as exc:
        _logger.error("创建或更新用户失败: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
