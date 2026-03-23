# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Configuration
APP_NAME = "AIMaxiFAI"
BASE_DIR = os.path.abspath(os.getcwd())

# Helper to collect all required modules
def collect_required_modules():
    additional_modules = [
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        "fastapi",
        "app",
    ]
    # Collect local submodules
    local_submodules = (
        collect_submodules("core") +
        collect_submodules("routers") +
        collect_submodules("endpoints") +
        collect_submodules("engine") +
        collect_submodules("services") +
        collect_submodules("models") +
        collect_submodules("streaming") +
        collect_submodules("webrtc")
    )
    return additional_modules + local_submodules

# Asset collection
def get_asset_files():
    assets = [
        ("config.yaml", "."),
        ("assets/voice_messages/Tone_B/Corrections", "assets/voice_messages/Tone_B/Corrections"),
    ]
    
    # Explicitly adding requested models
    models = [
        "yolo11s-seg.pt",
        "yolo11x-pose.pt",
        "pose_landmarker_heavy.task"
    ]
    for model in models:
        model_path = os.path.join("assets", "models", model)
        if os.path.exists(model_path):
            assets.append((model_path, os.path.join("assets", "models")))

    # Specifically ensuring Corrections is included if nested curiously
    corrections_path = os.path.join("assets", "voice_messages", "Tone_B", "Corrections")
    if os.path.exists(corrections_path):
        assets.append((corrections_path, corrections_path))
        
    return assets

hidden_imports = collect_required_modules()
asset_files = get_asset_files()

a = Analysis(
    ["start_api.py"],
    pathex=[BASE_DIR],
    binaries=[],
    datas=asset_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "IPython"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME,
)
