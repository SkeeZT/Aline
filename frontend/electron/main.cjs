const { app, BrowserWindow, dialog, protocol, ipcMain } = require("electron");
const path = require("path");

try {
  if (typeof app.setName === "function") {
    app.setName("AI MaxiFAI");
  }
} catch {}

const logger = require("./logger.cjs");
const BackendService = require("./backendService.cjs");
const fs = require("fs");

const backendService = new BackendService();
let mainWindow = null;
let splashWindow = null;

logger.log("Application starting...");

ipcMain.handle("restart-app", () => {
  app.relaunch();
  app.exit();
});

ipcMain.handle("get-app-version", () => app.getVersion());
ipcMain.handle("get-user-data-path", () => app.getPath("userData"));

function createSplashWindow() {
  logger.log("Creating splash window...");

  const isPackaged = app.isPackaged;
  const iconPath = path.join(__dirname, "../build/icon.ico");

  splashWindow = new BrowserWindow({
    width: 400,
    height: 400,
    transparent: false,
    icon: iconPath,
    frame: false,
    resizable: false,
    center: true,
    alwaysOnTop: true,
    skipTaskbar: true,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  const splashPath = !isPackaged
    ? path.join(app.getAppPath(), "public", "splash.html")
    : path.join(__dirname, "../dist", "splash.html");

  logger.log(`Loading splash screen from: ${splashPath}`);

  splashWindow.loadFile(splashPath);
  splashWindow.on("closed", () => {
    splashWindow = null;
  });

  splashWindow.setMenuBarVisibility(false);

  return splashWindow;
}

function createWindow() {
  logger.log("Creating main window...");

  const isPackaged = app.isPackaged;
  const iconPath = path.join(__dirname, "../build/icon.ico");

  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    show: false,
    icon: iconPath,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, "preload.cjs"),
    },
  });

  if (!isPackaged) {
    mainWindow.loadURL("http://localhost:5173");
    mainWindow.webContents.openDevTools();
  } else {
    const indexPath = path.join(__dirname, "../dist/index.html");
    mainWindow.loadFile(indexPath);
  }

  mainWindow.once("ready-to-show", () => {
    logger.log("Main window ready to show");
    if (splashWindow && !splashWindow.isDestroyed()) {
      splashWindow.close();
    }
    mainWindow.maximize();
    mainWindow.show();
  });
}

app.whenReady().then(async () => {
  try {
    if (process.platform === "win32") {
      app.setAppUserModelId("com.aimaxifai.app");
    }

    createSplashWindow();

    protocol.registerFileProtocol("file", (request, callback) => {
      let filePath = request.url.replace("file:///", "");
      if (process.platform === "win32") {
        filePath = filePath.startsWith("/") ? filePath.slice(1) : filePath;
      }
      try {
        const resolvedPath = path.normalize(decodeURIComponent(filePath));
        callback({ path: resolvedPath });
      } catch (error) {
        callback({ error: -2 });
      }
    });

    try {
      await backendService.start();
      logger.log("Backend service started successfully");
    } catch (error) {
      logger.error(`Backend service failed to start: ${error.message}`);
      dialog.showErrorBox(
        "Startup Error",
        `Failed to start the backend service.\n\nError: ${error.message}`
      );
      if (splashWindow && !splashWindow.isDestroyed()) {
        splashWindow.close();
      }
      app.quit();
      return;
    }

    createWindow();
  } catch (error) {
    logger.error(`Failed to start application: ${error.message}`);
    app.quit();
  }
});

app.on("window-all-closed", () => {
  backendService.stop();
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

app.on("before-quit", async () => {
  if (backendService) {
    backendService.stop();
  }
});
