/**
 * API utility for AI Trainer application
 * Handles communication with the backend API
 */

// Determine the base URL based on environment
const getBaseUrl = (): string => {
    if (import.meta.env.DEV) {
        // In development, use the Vite proxy or direct backend URL
        return 'http://localhost:8000';
    } else if (window.location.origin && window.location.origin.includes('localhost')) {
        // If frontend is served from localhost (during development)
        return 'http://localhost:8000';
    } else {
        // In packaged Electron app, backend runs on localhost:8000
        return 'http://localhost:8000';
    }
};

export const API_BASE_URL = getBaseUrl();

// Config Types

export interface DualCameraConfig {
    enabled: boolean;
    side_camera_id: number;
    front_camera_id: number;
    use_gstreamer: boolean;
    sync_tolerance_ms: number;
}

export interface ExerciseConfig {
    available_exercises: string[];
    default_exercise: string;
    experience_levels: string[];
    default_experience_level: string;
}

export interface AppConfig {
    dual_camera: DualCameraConfig;
    exercises: ExerciseConfig;
    api_version: string;
}

// Cached config to avoid repeated API calls
let cachedConfig: AppConfig | null = null;

/**
 * Fetches the application configuration from the backend.
 * Results are cached for the session.
 */
export const getAppConfig = async (forceRefresh = false): Promise<AppConfig> => {
    if (cachedConfig && !forceRefresh) {
        return cachedConfig;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/api/config/`);
        if (!response.ok) {
            throw new Error(`Failed to fetch config: ${response.status}`);
        }
        cachedConfig = await response.json();
        return cachedConfig!;
    } catch (error) {
        console.error('Failed to fetch app config:', error);
        // Return safe defaults
        return {
            dual_camera: {
                enabled: false,
                side_camera_id: 0,
                front_camera_id: 1,
                use_gstreamer: false,
                sync_tolerance_ms: 50,
            },
            exercises: {
                available_exercises: ['squat', 'bench_press'],
                default_exercise: 'squat',
                experience_levels: ['beginner', 'intermediate', 'advanced'],
                default_experience_level: 'intermediate',
            },
            api_version: '1.0.0',
        };
    }
};

/**
 * Fetches just the dual camera configuration.
 */
export const getDualCameraConfig = async (): Promise<DualCameraConfig> => {
    try {
        const response = await fetch(`${API_BASE_URL}/api/config/dual-camera`);
        if (!response.ok) {
            throw new Error(`Failed to fetch dual camera config: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error('Failed to fetch dual camera config:', error);
        return {
            enabled: false,
            side_camera_id: 0,
            front_camera_id: 1,
            use_gstreamer: false,
            sync_tolerance_ms: 50,
        };
    }
};

/**
 * Checks if dual camera mode is available (enabled in backend config).
 */
export const isDualCameraAvailable = async (): Promise<boolean> => {
    const config = await getDualCameraConfig();
    return config.enabled;
};

/**
 * Makes a request to the API
 * @param {string} endpoint - The API endpoint to call
import { API_BASE_URL, apiRequest } from '../utils/api';
 * @param {RequestInit} options - Request options (method, body, etc.)
 * @returns {Promise<Response>} - The response from the API
 */
export const apiRequest = async (endpoint: string, options: RequestInit = {}): Promise<Response> => {
    const url = `${API_BASE_URL}${endpoint}`;

    const headers: HeadersInit = {
        'Content-Type': 'application/json',
        ...(options.headers || {}),
    };

    const config: RequestInit = {
        ...options,
        headers,
    };

    // For multipart/form-data (file uploads), don't set Content-Type header
    if (options.body instanceof FormData) {
        // In fetch, if body is FormData, the browser sets Content-Type with boundary automatically
        // If we manually set strict types, we might need to cast headers
        // But simply deleting it from the headers object is the JS way.
        // In TS with RequestInit, headers is HeadersInit.
        // Using a type assertion or flexible object helps.
        const h = config.headers as Record<string, string>;
        if (h && h['Content-Type']) {
            delete h['Content-Type'];
        }
    }

    try {
        const response = await fetch(url, config);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return response;
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
};
