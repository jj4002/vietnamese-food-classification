#!/usr/bin/env python3
"""
Khởi động FastAPI server cho Vietnamese Food Classification Demo.
Chạy: python run_server.py
"""
import os
import sys
import subprocess
from pathlib import Path

# Đảm bảo chạy từ thư mục backend
os.chdir(Path(__file__).parent)

subprocess.run(
    [
        sys.executable, "-m", "uvicorn",
        "main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
    ],
    check=True,
)
