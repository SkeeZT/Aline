const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    restartApp: () => ipcRenderer.invoke('restart-app'),
    getAppVersion: () => ipcRenderer.invoke('get-app-version'),
    getUserDataPath: () => ipcRenderer.invoke('get-user-data-path'),
    sendBackendCommand: (command) => ipcRenderer.invoke('backend-command', command),
    onBackendResponse: (callback) => ipcRenderer.on('backend-response', (event, ...args) => callback(...args)),
    onUpdateStatus: (callback) => ipcRenderer.on('update-status', (event, ...args) => callback(...args)),
});
