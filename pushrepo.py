#!/usr/bin/env python3

import os
import sys
import subprocess


def run(cmd):
    print("$", " ".join(cmd))
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n[ERROR] Команда завершилась с кодом {result.returncode}")
        sys.exit(result.returncode)


ROOT = os.getcwd()

print("=== Git Repo Push Tool ===")
print(f"Репозиторий: {ROOT}\n")

# Проверяем git
if subprocess.run(
    ["git", "--version"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
).returncode != 0:
    print("[ERROR] Git не установлен.")
    print("Установи: pkg install git")
    sys.exit(1)

# Инициализация репозитория
if not os.path.isdir(os.path.join(ROOT, ".git")):
    print("[INFO] Git-репозитория нет. Создаю...")
    run(["git", "init"])

# Показываем состояние
run(["git", "status", "--short"])

# Добавляем ВСЕ файлы и папки
run(["git", "add", "-A"])

# Проверяем, есть ли изменения
result = subprocess.run(
    ["git", "diff", "--cached", "--quiet"]
)

if result.returncode == 0:
    print("\n[INFO] Нет изменений для коммита.")
    sys.exit(0)

# Сообщение коммита
message = input("\nСообщение коммита: ").strip()

if not message:
    message = "Update repository"

run(["git", "commit", "-m", message])

# Проверяем remote
remotes = subprocess.run(
    ["git", "remote"],
    capture_output=True,
    text=True
).stdout.split()

if not remotes:
    print("\n[INFO] Remote не настроен.")
    url = input("URL GitHub-репозитория: ").strip()

    if not url:
        print("[ERROR] URL не указан.")
        sys.exit(1)

    run(["git", "remote", "add", "origin", url])

# Пуш
branch = subprocess.run(
    ["git", "branch", "--show-current"],
    capture_output=True,
    text=True
).stdout.strip()

if not branch:
    branch = "main"
    run(["git", "branch", "-M", branch])

print(f"\n[INFO] Push: origin/{branch}")
run(["git", "push", "-u", "origin", branch])

print("\n=== PUSH УСПЕШЕН ===")