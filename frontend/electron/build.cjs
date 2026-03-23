const { execSync } = require("child_process");
const fs = require("fs-extra");
const path = require("path");

const LOG_DIR = path.join(__dirname, "../build-logs");
if (!fs.existsSync(LOG_DIR)) {
  fs.mkdirSync(LOG_DIR, { recursive: true });
}
const LOG_FILE = path.join(
  LOG_DIR,
  `build-${new Date().toISOString().replace(/:/g, "-").replace(/\./g, "-")}.log`
);

function buildLogger(message) {
  try {
    fs.appendFileSync(LOG_FILE, `[${new Date().toISOString()}] ${message}\n`);
  } catch {}
  console.log(message);
}

async function build() {
  try {
    buildLogger("Starting AI MaxiFAI build process...");
    
    // Clean directories
    buildLogger("Cleaning directories...");
    await fs.emptyDir("./build");
    await fs.emptyDir("./dist");
    await fs.emptyDir("./backend");

    // Create required directories
    await fs.ensureDir("./build");
    await fs.ensureDir("./public");

    // Build frontend
    buildLogger("Building frontend...");
    execSync("npm run build", { stdio: "inherit" });

    // Copy splash to dist
    buildLogger("Copying splash screen to dist...");
    if (fs.existsSync("./public/splash.html")) {
        await fs.copy("./public/splash.html", "./dist/splash.html");
    }

    // Build backend
    buildLogger("Building backend...");
    execSync("cd ../backend && uv run pyinstaller AIMaxiFAI.spec --noconfirm", {
      stdio: "inherit",
    });

    // Copy backend build
    buildLogger("Copying backend build...");
    const backendSourcePath = "../backend/dist/AIMaxiFAI";
    const backendDestPath = "./backend";
    
    if (fs.existsSync(backendSourcePath)) {
        await fs.copy(backendSourcePath, backendDestPath, {
            overwrite: true,
            dereference: true,
        });
        buildLogger("Backend copied successfully");
    } else {
        throw new Error(`Backend build not found at ${backendSourcePath}`);
    }

    // Packaging with electron-builder
    buildLogger("Packaging application with electron-builder...");
    execSync("npx electron-builder --win --x64 --publish never", {
        stdio: "inherit",
        env: {
            ...process.env,
            ELECTRON_BUILDER_ALLOW_UNRESOLVED_DEPENDENCIES: "true"
        }
    });

    buildLogger("Build completed successfully!");
  } catch (error) {
    buildLogger(`Build failed: ${error.message}`);
    process.exit(1);
  }
}

build();
