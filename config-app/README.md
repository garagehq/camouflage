# Camouflage Config App

Desktop application for configuring the Camouflage hand tracking system. This is the primary interface for controlling camera settings, customizing the pie menu, and managing connected accounts.

## Prerequisites

- **Node.js** (v18+)
- **Rust** (via rustup)

### Install Rust (if needed)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Setup

```bash
npm install
```

## Development

```bash
npm run tauri dev
```

This launches the app in development mode with hot reload.

## Build

```bash
npm run tauri:build
```

Build output location varies by platform:
- **macOS**: `src-tauri/target/release/bundle/macos/Camouflage Config.app`
- **Windows**: `src-tauri/target/release/bundle/msi/` or `nsis/`
- **Linux**: `src-tauri/target/release/bundle/deb/` or `appimage/`

## Features

- **Pie Menu Editor**: Drag-and-drop interface to configure which widgets appear in the in-stream pie menu (Draw, Nerd Stats, Timer, etc.)
- **Account Connections**: Connect Google account for calendar widget (OAuth)
- **Settings**: Toggle DepthAI hardware mode, virtual camera, FPS display, mirror mode

## Architecture

- **Frontend**: React + TypeScript + Vite
- **Backend**: Rust + Tauri 2
- **Secure Storage**: OS keychain for OAuth tokens (macOS Keychain, Windows Credential Manager)
- **Config File**: `~/.camouflage/config.json` - shared with the Python hand tracker

## Project Structure

```
config-app/
├── src/                    # React frontend
│   ├── App.tsx             # Main app with tab navigation
│   └── components/
│       ├── PieMenuEditor.tsx   # Drag-and-drop pie menu config
│       ├── AccountsPanel.tsx   # OAuth account management
│       └── SettingsPanel.tsx   # General settings
├── src-tauri/              # Rust backend
│   ├── src/lib.rs          # Tauri commands (keychain, config)
│   └── tauri.conf.json     # App configuration
└── package.json
```
