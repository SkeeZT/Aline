
import cv2
import platform
import time
import subprocess
import json

def get_camera_names_windows():
    """
    Get camera names on Windows using PowerShell.
    Returns a list of names. The order might not strictly match OpenCV indices, 
    but usually standard PnP order is followed.
    """
    try:
        # PowerShell command to get Camera and Image devices
        cmd = "Get-PnpDevice -Class Camera,Image -Status OK | Select-Object FriendlyName | ConvertTo-Json"
        result = subprocess.run(["powershell", "-Command", cmd], capture_output=True, text=True)
        
        if result.returncode != 0:
            return []
            
        data = json.loads(result.stdout)
        
        # Normalize to list if single object
        if isinstance(data, dict):
            data = [data]
            
        names = []
        if isinstance(data, list):
            for device in data:
                name = device.get("FriendlyName", "")
                if name:
                    names.append(name)
        
        # Filter strictly reasonable names if needed, but usually these are the cameras
        return names
    except Exception as e:
        print(f"Could not fetch camera names: {e}")
        return []

def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    is_working = True
    dev_port = 0
    working_ports = []
    available_ports = []
    
    # Try to get names
    camera_names = []
    if platform.system() == "Windows":
        camera_names = get_camera_names_windows()
        print(f"Detected Camera Devices (Windows PnP): {camera_names}\n")
    
    print("Scanning for cameras using OpenCV... (this may take a few seconds)")

    # Scan up to 5 ports (reduced from 10 to check relevant ones faster)
    max_ports = 5 
    while dev_port < max_ports:
        # On Windows, using CAP_DSHOW is often faster/more reliable for index checking
        if platform.system() == "Windows":
             camera = cv2.VideoCapture(dev_port, cv2.CAP_DSHOW)
        else:
             camera = cv2.VideoCapture(dev_port)

        if camera.isOpened():
            is_reading, img = camera.read()
            w = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = camera.get(cv2.CAP_PROP_FPS)
            
            # Retrieve name if available (heuristic matching)
            guessed_name = "Unknown"
            if dev_port < len(camera_names):
                guessed_name = f"Possibly '{camera_names[dev_port]}'"
            
            if is_reading:
                print(f"Port {dev_port}: Working. Resolution: {int(w)}x{int(h)} @ {fps} FPS. {guessed_name}")
                working_ports.append(dev_port)
            else:
                print(f"Port {dev_port}: Available but failed to read frame. {guessed_name}")
                available_ports.append(dev_port)
            
            camera.release()
        else:
            # print(f"Port {dev_port}: Not detected")
            pass
            
        dev_port += 1
        
    return working_ports, available_ports

if __name__ == "__main__":
    print(f"OS: {platform.system()}")
    print("-----------")
    
    start_time = time.time()
    working, available = list_ports()
    end_time = time.time()
    
    print("-----------")
    if len(working) > 0:
        print(f"Found {len(working)} working camera(s) at indices: {working}")
        print("\nNote: 'Front' camera (index 0) is usually your built-in webcam.")
        print("'Side' camera (index 1) is usually your external USB camera.")
        print("Adjust indices in config.yaml if this order is different.")
    else:
        print("No working cameras found.")
        
    if len(available) > 0:
        print(f"Found {len(available)} other available device(s) at indices: {available}")
        
    print(f"\nScan completed in {end_time - start_time:.2f} seconds.")
