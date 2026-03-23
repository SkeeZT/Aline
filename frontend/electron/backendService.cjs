const { spawn } = require("child_process");
const path = require("path");
const { app, BrowserWindow, dialog } = require("electron");
const fs = require("fs");
const http = require("http");
const https = require("https");
const logger = require("./logger.cjs");
const config = require("./config.cjs");

class BackendService {
  constructor() {
    this.process = null;
    this.isRunning = false;
    this.retryCount = 0;
    this.maxRetries = 5;
    this.retryDelay = 3000;
    this.startupTimeout = 60000;
  }

  updateSplashStatus(message) {
    const allWindows = BrowserWindow.getAllWindows();
    const splashWindow = allWindows.find(
      (w) => w.webContents && w.webContents.getURL().includes("splash.html")
    );

    if (splashWindow && !splashWindow.isDestroyed()) {
      try {
        splashWindow.webContents.send("update-status", message);
      } catch (error) {
        logger.error(`Failed to update splash status: ${error.message}`);
      }
    }
  }

  async start() {
    if (this.isRunning) return;

    const isPackaged = app.isPackaged;
    const backendPath = isPackaged
      ? path.join(process.resourcesPath, "backend")
      : path.join(__dirname, "../../backend");

    logger.log(`Starting backend from: ${backendPath}`);

    const executable = isPackaged ? path.join(backendPath, "AIMaxiFAI.exe") : "python";
    const args = isPackaged
      ? ["--host", "127.0.0.1", "--port", config.backend.port]
      : ["start_api.py", "--host", "127.0.0.1", "--port", config.backend.port];

    const env = {
      ...process.env,
      PYTHONUNBUFFERED: "1",
      NODE_ENV: isPackaged ? "production" : "development",
      PATH: isPackaged ? `${backendPath};${process.env.PATH}` : process.env.PATH
    };

    const spawnOptions = {
        cwd: backendPath,
        stdio: ["ignore", "pipe", "pipe"],
        env,
        windowsHide: true,
        shell: false
    };

    this.updateSplashStatus("Starting backend service...");
    this.process = spawn(executable, args, spawnOptions);

    this.process.stdout.on("data", (data) => {
      logger.log(`Backend: ${data.toString().trim()}`);
    });

    this.process.stderr.on("data", (data) => {
      logger.error(`Backend Error: ${data.toString().trim()}`);
    });

    this.process.on("exit", (code) => {
      logger.log(`Backend process exited with code ${code}`);
      this.isRunning = false;
    });

    await this.waitForBackend();
    this.isRunning = true;
  }

  async waitForBackend() {
    const maxAttempts = 30;
    const delay = 2000;
    let attempts = 0;

    const healthUrl = `${config.backend.url}/health`; // Or whatever health endpoint you have

    while (attempts < maxAttempts) {
      try {
        await new Promise((resolve, reject) => {
          const req = http.get(healthUrl, (res) => {
            if (res.statusCode === 200) resolve();
            else reject();
          });
          req.on("error", reject);
          req.end();
        });
        logger.log("Backend health check passed");
        return;
      } catch (error) {
        attempts++;
        if (attempts % 5 === 0) {
          this.updateSplashStatus(`Waiting for backend... (${attempts}/${maxAttempts})`);
        }
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
    throw new Error("Backend failed to start within timeout");
  }

  stop() {
    if (this.process) {
      this.process.kill();
      this.process = null;
      this.isRunning = false;
    }
  }
}

module.exports = BackendService;
