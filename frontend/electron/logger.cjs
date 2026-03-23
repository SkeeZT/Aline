const fs = require('fs');
const path = require('path');
const { app } = require('electron');

const logPath = path.join(app.getPath('userData'), 'app.log');

function log(message) {
    const timestamp = new Date().toISOString();
    const formattedMessage = `[${timestamp}] ${message}\n`;
    console.log(message);
    try {
        fs.appendFileSync(logPath, formattedMessage);
    } catch (err) {
        console.error('Failed to write to log file', err);
    }
}

function error(message) {
    log(`ERROR: ${message}`);
}

module.exports = {
    log,
    error,
    logPath
};
