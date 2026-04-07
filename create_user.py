# create_user.py

import os
import json
import getpass
import hashlib
import secrets
import traceback


USERS_FILE = "config/users.json"
PBKDF2_ITERATIONS = 200000


def log(msg: str):
    print(f"[INFO] {msg}")


def load_users():
    try:
        if not os.path.exists(USERS_FILE):
            log(f"用户文件不存在，将自动创建: {USERS_FILE}")
            return []

        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            log("用户文件格式错误，已重置为空列表。")
            return []

        return data

    except Exception as e:
        log(f"加载用户文件失败: {e}")
        traceback.print_exc()
        return []


def save_users(users):
    try:
        os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)

        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)

        log(f"用户文件保存成功: {USERS_FILE}")

    except Exception as e:
        log(f"保存用户文件失败: {e}")
        traceback.print_exc()
        raise


def make_password_record(password: str) -> dict:
    """
    使用 PBKDF2-HMAC-SHA256 生成密码记录。
    返回：
    {
        "salt": "...",
        "password_hash": "...",
        "hash_method": "pbkdf2_sha256",
        "iterations": 200000
    }
    """
    try:
        salt = secrets.token_bytes(16)
        pwd_bytes = password.encode("utf-8")

        dk = hashlib.pbkdf2_hmac(
            "sha256",
            pwd_bytes,
            salt,
            PBKDF2_ITERATIONS
        )

        return {
            "salt": salt.hex(),
            "password_hash": dk.hex(),
            "hash_method": "pbkdf2_sha256",
            "iterations": PBKDF2_ITERATIONS
        }

    except Exception as e:
        log(f"生成密码记录失败: {e}")
        traceback.print_exc()
        raise


def find_user(users, username: str):
    for user in users:
        if isinstance(user, dict) and user.get("username") == username:
            return user
    return None


def main():
    print("=" * 70)
    print("本地用户创建工具（PBKDF2 版本）")
    print("=" * 70)

    try:
        username = input("请输入用户名: ").strip()
        if not username:
            print("[ERROR] 用户名不能为空。")
            return

        password = getpass.getpass("请输入密码: ").strip()
        if not password:
            print("[ERROR] 密码不能为空。")
            return

        confirm_password = getpass.getpass("请再次输入密码确认: ").strip()
        if password != confirm_password:
            print("[ERROR] 两次输入的密码不一致。")
            return

        users = load_users()
        password_record = make_password_record(password)

        existing_user = find_user(users, username)

        if existing_user:
            existing_user["salt"] = password_record["salt"]
            existing_user["password_hash"] = password_record["password_hash"]
            existing_user["hash_method"] = password_record["hash_method"]
            existing_user["iterations"] = password_record["iterations"]
            print(f"[INFO] 用户已存在，密码已更新: {username}")
        else:
            users.append({
                "username": username,
                "salt": password_record["salt"],
                "password_hash": password_record["password_hash"],
                "hash_method": password_record["hash_method"],
                "iterations": password_record["iterations"]
            })
            print(f"[INFO] 新用户创建成功: {username}")

        save_users(users)

        print("\n[OK] 操作完成。当前 users.json 内容如下：")
        print(json.dumps(users, ensure_ascii=False, indent=2))

    except KeyboardInterrupt:
        print("\n[WARN] 用户取消操作。")

    except Exception as e:
        print(f"[ERROR] 创建用户失败: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()