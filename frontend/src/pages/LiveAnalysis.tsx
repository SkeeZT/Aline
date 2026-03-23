import React, { useEffect, useRef, useState } from 'react';
import { Box, Typography, Button, Card, Grid, Alert, Chip, Stack, Paper, FormControl, InputLabel, Select, MenuItem, Switch, FormControlLabel, Tabs, Tab, CircularProgress } from '@mui/material';
import type { SelectChangeEvent } from '@mui/material';
import { Camera, StopCircle, CheckCircle2, MonitorPlay, Maximize2, AlertCircle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { getAppConfig, type AppConfig } from '../utils/api';

// Exercise options
const EXERCISES = [
    { value: 'squat', label: 'Squat', icon: '🏋️', dualCameraRecommended: true },
    { value: 'pullup', label: 'Pull-up', icon: '💪', dualCameraRecommended: true },
    { value: 'pushup', label: 'Push-up', icon: '🙌', dualCameraRecommended: true },
    { value: 'dips', label: 'Dips', icon: '⬇️', dualCameraRecommended: true },
    { value: 'lunges', label: 'Lunges', icon: '🦵', dualCameraRecommended: true },
    { value: 'plank', label: 'Plank', icon: '🧘', dualCameraRecommended: false },
    { value: 'deadlift', label: 'Deadlift', icon: '🏋️‍♂️', dualCameraRecommended: true },
    { value: 'overhead_press', label: 'Overhead Press', icon: '🙆', dualCameraRecommended: true },
    { value: 'bent_over_row', label: 'Bent-Over Row', icon: '🚣', dualCameraRecommended: true },
    { value: 'glute_bridge', label: 'Glute Bridge', icon: '🌉', dualCameraRecommended: false },
    { value: 'wall_sit', label: 'Wall Sit', icon: '🧱', dualCameraRecommended: false },
    { value: 'bench_press', label: 'Bench Press', icon: '🏋️‍♂️', dualCameraRecommended: true },
];

const EXPERIENCE_LEVELS = [
    { value: 'beginner', label: 'Beginner' },
    { value: 'intermediate', label: 'Intermediate' },
    { value: 'advanced', label: 'Advanced' },
];

// Camera view modes for dual camera
type ViewMode = 'grid' | 'front' | 'side' | 'combined';

export const LiveAnalysis: React.FC = () => {
    const navigate = useNavigate();

    // Single camera refs
    const rawCanvasRef = useRef<HTMLCanvasElement>(null);
    const processedCanvasRef = useRef<HTMLCanvasElement>(null);

    // Dual camera refs
    const frontRawCanvasRef = useRef<HTMLCanvasElement>(null);
    const frontProcessedCanvasRef = useRef<HTMLCanvasElement>(null);
    const sideRawCanvasRef = useRef<HTMLCanvasElement>(null);
    const sideProcessedCanvasRef = useRef<HTMLCanvasElement>(null);
    const combinedCanvasRef = useRef<HTMLCanvasElement>(null);

    const wsRef = useRef<WebSocket | null>(null);
    const firstFrameReceived = useRef(false);

    const [isStreaming, setIsStreaming] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [status, setStatus] = useState<string>('Ready');
    const [targetReps, setTargetReps] = useState<number | null>(10);
    const [currentReps, setCurrentReps] = useState(0);
    const [exerciseType, setExerciseType] = useState<string>('squat');
    const [experienceLevel, setExperienceLevel] = useState<string>('intermediate');

    // Dual camera state
    const [dualCameraMode, setDualCameraMode] = useState(false);
    const [viewMode, setViewMode] = useState<ViewMode>('grid');
    const [cameraStatus, setCameraStatus] = useState<{ front: boolean; side: boolean }>({ front: false, side: false });

    // Config state - tracks whether dual camera is available in backend
    const [appConfig, setAppConfig] = useState<AppConfig | null>(null);
    const [configLoading, setConfigLoading] = useState(true);
    const dualCameraAvailable = appConfig?.dual_camera?.enabled ?? false;

    // Load config on mount
    useEffect(() => {
        const loadConfig = async () => {
            try {
                setConfigLoading(true);
                const config = await getAppConfig();
                setAppConfig(config);

                // Set default mode based on config
                if (config.dual_camera.enabled) {
                    setDualCameraMode(true);
                } else {
                    setDualCameraMode(false);
                }
            } catch (err) {
                console.error('Failed to load config:', err);
                // Default to single camera mode on error
                setDualCameraMode(false);
            } finally {
                setConfigLoading(false);
            }
        };

        loadConfig();

        return () => {
            stopStream();
        };
    }, []);

    // Helper to draw frame on canvas
    const drawFrame = (dataUrl: string, canvasRef: React.RefObject<HTMLCanvasElement | null>) => {
        if (!canvasRef.current) return;
        const img = new Image();
        img.onload = () => {
            const ctx = canvasRef.current?.getContext('2d');
            if (ctx && canvasRef.current) {
                ctx.drawImage(img, 0, 0, canvasRef.current.width, canvasRef.current.height);
            }
        };
        img.src = dataUrl;
    };

    // Helper to clear canvas
    const clearCanvas = (canvasRef: React.RefObject<HTMLCanvasElement | null>) => {
        if (canvasRef.current) {
            const ctx = canvasRef.current.getContext('2d');
            if (ctx) {
                ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
            }
        }
    };

    const startStream = async () => {
        try {
            firstFrameReceived.current = false; // Reset flag
            setIsLoading(true);
            setError(null);
            setCameraStatus({ front: false, side: false });

            // Build WebSocket URL with all parameters
            const params = new URLSearchParams();
            if (targetReps) params.append('target_reps', targetReps.toString());
            params.append('exercise_type', exerciseType);
            params.append('experience_level', experienceLevel);
            params.append('dual_camera', dualCameraMode.toString());

            const wsUrl = `ws://${window.location.hostname}:8000/ws/analyze?${params.toString()}`;
            const ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                console.log(`Connected to server for ${exerciseType} (${dualCameraMode ? 'dual' : 'single'} camera) - waiting for camera...`);
                // Send config as backup method (URL params are primary)
                ws.send(JSON.stringify({
                    target_reps: targetReps,
                    exercise_type: exerciseType,
                    experience_level: experienceLevel,
                    dual_camera: dualCameraMode
                }));
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);

                // Handle ready message from backend
                if (data.ready) {
                    console.log('Camera ready:', data.message);
                    if (data.camera_status) {
                        setCameraStatus(data.camera_status);
                    }
                    return;
                }

                // First frame received - camera is ready, hide loading overlay
                const hasFrame = dualCameraMode
                    ? (data.front_raw || data.side_raw || data.combined_frame)
                    : (data.raw_frame || data.processed_frame);

                if (!firstFrameReceived.current && hasFrame) {
                    console.log('First frame received - hiding loading overlay');
                    firstFrameReceived.current = true;
                    setIsLoading(false);
                    setIsStreaming(true);
                }

                if (data.metadata && data.metadata.state) {
                    setStatus(data.metadata.state.toUpperCase());
                }

                // Update camera status if provided
                if (data.camera_status) {
                    setCameraStatus(data.camera_status);
                }

                // Handle Redirect
                if (data.metadata && data.metadata.redirect) {
                    console.log('Redirecting to results:', data.metadata.filename);
                    stopStream(); // Ensure cleanup
                    navigate('/analysis/results', {
                        state: { filename: data.metadata.filename }
                    });
                    return;
                }

                // Update rep count
                if (data.metadata) {
                    if (data.metadata.total_reps !== undefined) {
                        setCurrentReps(data.metadata.total_reps);
                    }

                    // Check if session completed
                    if (data.metadata.completed) {
                        if (data.metadata.reason === "target_reps_reached") {
                            setStatus(`COMPLETED - ${data.metadata.total_reps} reps reached!`);
                            setTimeout(() => {
                                stopStream();
                            }, 2000);
                        }
                    }
                }

                if (dualCameraMode) {
                    // Dual camera mode: handle front/side frames
                    if (data.front_raw) drawFrame(data.front_raw, frontRawCanvasRef);
                    if (data.front_processed) drawFrame(data.front_processed, frontProcessedCanvasRef);
                    if (data.side_raw) drawFrame(data.side_raw, sideRawCanvasRef);
                    if (data.side_processed) drawFrame(data.side_processed, sideProcessedCanvasRef);
                    if (data.combined_frame) drawFrame(data.combined_frame, combinedCanvasRef);
                } else {
                    // Single camera mode
                    if (data.raw_frame) drawFrame(data.raw_frame, rawCanvasRef);
                    if (data.processed_frame) drawFrame(data.processed_frame, processedCanvasRef);
                }
            };

            ws.onerror = (err) => {
                console.error('WebSocket error:', err);
                setError('Connection error. Is the backend running?');
                setIsLoading(false);
                stopStream();
            };

            ws.onclose = () => {
                console.log('Disconnected');
                setIsLoading(false);

                // Clear all canvases when connection closes
                if (dualCameraMode) {
                    clearCanvas(frontRawCanvasRef);
                    clearCanvas(frontProcessedCanvasRef);
                    clearCanvas(sideRawCanvasRef);
                    clearCanvas(sideProcessedCanvasRef);
                    clearCanvas(combinedCanvasRef);
                } else {
                    clearCanvas(rawCanvasRef);
                    clearCanvas(processedCanvasRef);
                }

                setIsStreaming(false);
                setCameraStatus({ front: false, side: false });
            };

            wsRef.current = ws;

        } catch (err) {
            console.error('Error connecting to server:', err);
            setError('Could not connect to server. Is the backend running?');
            setIsLoading(false);
        }
    };

    const stopStream = () => {
        if (wsRef.current) {
            // Send stop command to backend
            if (wsRef.current.readyState === WebSocket.OPEN) {
                wsRef.current.send('stop');
            }
            wsRef.current.close();
            wsRef.current = null;
        }

        // Clear all canvases
        clearCanvas(rawCanvasRef);
        clearCanvas(processedCanvasRef);
        clearCanvas(frontRawCanvasRef);
        clearCanvas(frontProcessedCanvasRef);
        clearCanvas(sideRawCanvasRef);
        clearCanvas(sideProcessedCanvasRef);
        clearCanvas(combinedCanvasRef);

        setIsStreaming(false);
        setStatus('Ready');
        setCameraStatus({ front: false, side: false });
    };

    // Get current exercise info
    const currentExercise = EXERCISES.find(e => e.value === exerciseType);

    return (
        <Box sx={{ maxWidth: dualCameraMode ? 'xl' : 'lg', mx: 'auto' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4, flexWrap: 'wrap', gap: 2 }}>
                <Box>
                    <Typography variant="h4" fontWeight="bold" gutterBottom>
                        Live Analysis
                    </Typography>
                    <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
                        <Typography variant="body2" color="text.secondary">Status:</Typography>
                        <Chip
                            label={status}
                            color={status === 'EXERCISING' ? 'success' : status === 'Ready' ? 'default' : 'primary'}
                            size="small"
                            variant="outlined"
                        />
                        {dualCameraMode && isStreaming && (
                            <>
                                <Chip
                                    label={`Front: ${cameraStatus.front ? 'OK' : 'OFF'}`}
                                    color={cameraStatus.front ? 'success' : 'error'}
                                    size="small"
                                    variant="outlined"
                                />
                                <Chip
                                    label={`Side: ${cameraStatus.side ? 'OK' : 'OFF'}`}
                                    color={cameraStatus.side ? 'success' : 'error'}
                                    size="small"
                                    variant="outlined"
                                />
                            </>
                        )}
                    </Stack>
                </Box>
                <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
                    {configLoading && (
                        <CircularProgress size={20} sx={{ mr: 2 }} />
                    )}
                    {!isStreaming && !configLoading && (
                        <>
                            {/* Dual Camera Toggle - Only show if available in config */}
                            {dualCameraAvailable ? (
                                <FormControlLabel
                                    control={
                                        <Switch
                                            checked={dualCameraMode}
                                            onChange={(e) => setDualCameraMode(e.target.checked)}
                                            color="primary"
                                        />
                                    }
                                    label={
                                        <Stack direction="row" spacing={0.5} alignItems="center">
                                            <MonitorPlay size={16} />
                                            <Typography variant="body2">Dual Camera</Typography>
                                        </Stack>
                                    }
                                    sx={{ mr: 1 }}
                                />
                            ) : (
                                <Chip
                                    icon={<AlertCircle size={14} />}
                                    label="Single Camera Mode"
                                    size="small"
                                    variant="outlined"
                                    sx={{ mr: 1 }}
                                />
                            )}

                            {/* Exercise Type Selector */}
                            <FormControl size="small" sx={{ minWidth: 140 }}>
                                <InputLabel id="exercise-select-label">Exercise</InputLabel>
                                <Select
                                    labelId="exercise-select-label"
                                    value={exerciseType}
                                    label="Exercise"
                                    onChange={(e: SelectChangeEvent) => setExerciseType(e.target.value)}
                                >
                                    {EXERCISES.map((ex) => (
                                        <MenuItem key={ex.value} value={ex.value}>
                                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                <span>{ex.icon}</span>
                                                <span>{ex.label}</span>
                                                {ex.dualCameraRecommended && dualCameraMode && (
                                                    <Chip label="✓" size="small" color="success" sx={{ height: 18, fontSize: '0.65rem' }} />
                                                )}
                                            </Box>
                                        </MenuItem>
                                    ))}
                                </Select>
                            </FormControl>

                            {/* Experience Level Selector */}
                            <FormControl size="small" sx={{ minWidth: 130 }}>
                                <InputLabel id="level-select-label">Level</InputLabel>
                                <Select
                                    labelId="level-select-label"
                                    value={experienceLevel}
                                    label="Level"
                                    onChange={(e: SelectChangeEvent) => setExperienceLevel(e.target.value)}
                                >
                                    {EXPERIENCE_LEVELS.map((level) => (
                                        <MenuItem key={level.value} value={level.value}>
                                            {level.label}
                                        </MenuItem>
                                    ))}
                                </Select>
                            </FormControl>

                            {/* Target Reps */}
                            <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                                <Typography variant="body2" color="text.secondary">
                                    Reps:
                                </Typography>
                                <input
                                    type="number"
                                    min="1"
                                    max="100"
                                    value={targetReps || ''}
                                    onChange={(e) => setTargetReps(e.target.value ? parseInt(e.target.value) : null)}
                                    style={{
                                        width: '60px',
                                        padding: '4px 8px',
                                        borderRadius: '4px',
                                        border: '1px solid rgba(255,255,255,0.2)',
                                        background: 'rgba(255,255,255,0.1)',
                                        color: 'white'
                                    }}
                                    placeholder="∞"
                                />
                            </Box>
                        </>
                    )}
                    {!isStreaming ? (
                        <Button
                            variant="contained"
                            startIcon={<Camera />}
                            onClick={startStream}
                            size="large"
                        >
                            {dualCameraMode ? 'Start Cameras' : 'Start Camera'}
                        </Button>
                    ) : (
                        <Button
                            variant="outlined"
                            color="error"
                            startIcon={<StopCircle />}
                            onClick={stopStream}
                            size="large"
                        >
                            Stop Session
                        </Button>
                    )}
                </Box>
            </Box>

            {error && (
                <Alert severity="error" sx={{ mb: 3 }}>
                    {error}
                </Alert>
            )}

            {isLoading && (
                <Box
                    sx={{
                        position: 'fixed',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        bgcolor: 'rgba(0, 0, 0, 0.8)',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        zIndex: 9999,
                    }}
                >
                    <Box sx={{ textAlign: 'center' }}>
                        <Box
                            sx={{
                                width: 50,
                                height: 50,
                                border: '4px solid rgba(255, 255, 255, 0.3)',
                                borderTop: '4px solid #60a5fa',
                                borderRadius: '50%',
                                animation: 'spin 1s linear infinite',
                                mx: 'auto',
                                mb: 3,
                            }}
                        />
                        <Typography variant="h6" color="white" gutterBottom>
                            Initializing Camera...
                        </Typography>
                        <Typography variant="body2" color="rgba(255, 255, 255, 0.7)">
                            Loading AI models and opening camera
                        </Typography>
                    </Box>
                </Box>
            )}

            <style>{`
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            `}</style>

            <Grid container spacing={4}>
                <Grid size={{ xs: 12, lg: dualCameraMode ? 9 : 8 }}>
                    {dualCameraMode ? (
                        /* Dual Camera Mode UI */
                        <Box>
                            {/* View Mode Tabs */}
                            <Tabs
                                value={viewMode}
                                onChange={(_, newValue) => setViewMode(newValue as ViewMode)}
                                sx={{ mb: 2 }}
                                centered
                            >
                                <Tab value="grid" label="Grid View" icon={<Maximize2 size={16} />} iconPosition="start" />
                                <Tab value="combined" label="Combined" icon={<MonitorPlay size={16} />} iconPosition="start" />
                                <Tab value="front" label="Front Camera" />
                                <Tab value="side" label="Side Camera" />
                            </Tabs>

                            {viewMode === 'grid' && (
                                <Grid container spacing={2}>
                                    {/* Front Processed */}
                                    <Grid size={{ xs: 12, md: 6 }}>
                                        <Typography variant="subtitle1" align="center" display="block" gutterBottom sx={{ color: 'success.main', fontWeight: 'bold' }}>
                                            Front Analysis
                                        </Typography>
                                        <Paper
                                            elevation={0}
                                            sx={{
                                                p: 0.5,
                                                bgcolor: 'background.paper',
                                                borderRadius: 2,
                                                overflow: 'hidden',
                                                position: 'relative',
                                                aspectRatio: '3/4',
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                                border: '2px solid rgba(74, 222, 128, 0.5)'
                                            }}
                                        >
                                            <canvas
                                                ref={frontProcessedCanvasRef}
                                                width={480}
                                                height={640}
                                                style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                                            />
                                            {!isStreaming && (
                                                <Box sx={{ position: 'absolute', textAlign: 'center', opacity: 0.5 }}>
                                                    <Camera size={32} />
                                                    <Typography variant="caption" display="block">Front Feed</Typography>
                                                </Box>
                                            )}
                                        </Paper>
                                    </Grid>

                                    {/* Side Processed */}
                                    <Grid size={{ xs: 12, md: 6 }}>
                                        <Typography variant="subtitle1" align="center" display="block" gutterBottom sx={{ color: 'secondary.main', fontWeight: 'bold' }}>
                                            Side Analysis
                                        </Typography>
                                        <Paper
                                            elevation={0}
                                            sx={{
                                                p: 0.5,
                                                bgcolor: 'background.paper',
                                                borderRadius: 2,
                                                overflow: 'hidden',
                                                position: 'relative',
                                                aspectRatio: '3/4',
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                                border: '2px solid rgba(244, 114, 182, 0.5)'
                                            }}
                                        >
                                            <canvas
                                                ref={sideProcessedCanvasRef}
                                                width={480}
                                                height={640}
                                                style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                                            />
                                            {!isStreaming && (
                                                <Box sx={{ position: 'absolute', textAlign: 'center', opacity: 0.5 }}>
                                                    <Camera size={32} />
                                                    <Typography variant="caption" display="block">Side Feed</Typography>
                                                </Box>
                                            )}
                                        </Paper>
                                    </Grid>
                                </Grid>
                            )}

                            {/* Combined View */}
                            {viewMode === 'combined' && (
                                <Box sx={{ display: 'flex', justifyContent: 'center' }}>
                                    <Paper
                                        elevation={0}
                                        sx={{
                                            p: 1,
                                            bgcolor: 'background.paper',
                                            borderRadius: 4,
                                            overflow: 'hidden',
                                            position: 'relative',
                                            width: '100%',
                                            maxWidth: 900,
                                            aspectRatio: '16/9',
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'center',
                                            border: '2px solid rgba(96, 165, 250, 0.3)'
                                        }}
                                    >
                                        <canvas
                                            ref={combinedCanvasRef}
                                            width={1280}
                                            height={720}
                                            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                                        />
                                        {!isStreaming && (
                                            <Box sx={{ position: 'absolute', textAlign: 'center', opacity: 0.5 }}>
                                                <MonitorPlay size={64} style={{ marginBottom: 16 }} />
                                                <Typography>Combined Analysis View</Typography>
                                            </Box>
                                        )}
                                    </Paper>
                                </Box>
                            )}

                            {/* Front Camera Only View */}
                            {viewMode === 'front' && (
                                <Grid container spacing={2}>
                                    <Grid size={{ xs: 12, md: 6 }}>
                                        <Typography variant="subtitle1" align="center" gutterBottom sx={{ color: 'primary.main' }}>
                                            Front Raw Feed
                                        </Typography>
                                        <Paper
                                            elevation={0}
                                            sx={{
                                                p: 1,
                                                bgcolor: 'background.paper',
                                                borderRadius: 4,
                                                overflow: 'hidden',
                                                position: 'relative',
                                                aspectRatio: '3/4',
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                                border: '1px solid rgba(96, 165, 250, 0.3)'
                                            }}
                                        >
                                            <canvas
                                                ref={frontRawCanvasRef}
                                                width={480}
                                                height={640}
                                                style={{ width: '100%', height: '100%', objectFit: 'contain', transform: 'scaleX(-1)' }}
                                            />
                                        </Paper>
                                    </Grid>
                                    <Grid size={{ xs: 12, md: 6 }}>
                                        <Typography variant="subtitle1" align="center" gutterBottom sx={{ color: 'success.main' }}>
                                            Front AI Analysis
                                        </Typography>
                                        <Paper
                                            elevation={0}
                                            sx={{
                                                p: 1,
                                                bgcolor: 'background.paper',
                                                borderRadius: 4,
                                                overflow: 'hidden',
                                                position: 'relative',
                                                aspectRatio: '3/4',
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                                border: '1px solid rgba(74, 222, 128, 0.3)'
                                            }}
                                        >
                                            <canvas
                                                ref={frontProcessedCanvasRef}
                                                width={480}
                                                height={640}
                                                style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                                            />
                                        </Paper>
                                    </Grid>
                                </Grid>
                            )}

                            {/* Side Camera Only View */}
                            {viewMode === 'side' && (
                                <Grid container spacing={2}>
                                    <Grid size={{ xs: 12, md: 6 }}>
                                        <Typography variant="subtitle1" align="center" gutterBottom sx={{ color: 'warning.main' }}>
                                            Side Raw Feed
                                        </Typography>
                                        <Paper
                                            elevation={0}
                                            sx={{
                                                p: 1,
                                                bgcolor: 'background.paper',
                                                borderRadius: 4,
                                                overflow: 'hidden',
                                                position: 'relative',
                                                aspectRatio: '3/4',
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                                border: '1px solid rgba(251, 191, 36, 0.3)'
                                            }}
                                        >
                                            <canvas
                                                ref={sideRawCanvasRef}
                                                width={480}
                                                height={640}
                                                style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                                            />
                                        </Paper>
                                    </Grid>
                                    <Grid size={{ xs: 12, md: 6 }}>
                                        <Typography variant="subtitle1" align="center" gutterBottom sx={{ color: 'secondary.main' }}>
                                            Side AI Analysis
                                        </Typography>
                                        <Paper
                                            elevation={0}
                                            sx={{
                                                p: 1,
                                                bgcolor: 'background.paper',
                                                borderRadius: 4,
                                                overflow: 'hidden',
                                                position: 'relative',
                                                aspectRatio: '3/4',
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                                border: '1px solid rgba(244, 114, 182, 0.3)'
                                            }}
                                        >
                                            <canvas
                                                ref={sideProcessedCanvasRef}
                                                width={480}
                                                height={640}
                                                style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                                            />
                                        </Paper>
                                    </Grid>
                                </Grid>
                            )}
                        </Box>
                    ) : (
                        /* Single Camera Mode UI */
                        <Grid container spacing={2}>
                            {/* Raw Camera Feed */}
                            <Grid size={{ xs: 12, md: 6 }}>
                                <Typography variant="subtitle1" align="center" gutterBottom>Raw Feed</Typography>
                                <Paper
                                    elevation={0}
                                    sx={{
                                        p: 1,
                                        bgcolor: 'background.paper',
                                        borderRadius: 4,
                                        overflow: 'hidden',
                                        position: 'relative',
                                        aspectRatio: '3/4',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        border: '1px solid rgba(255,255,255,0.1)'
                                    }}
                                >
                                    <canvas
                                        ref={rawCanvasRef}
                                        width={480}
                                        height={640}
                                        style={{ width: '100%', height: '100%', objectFit: 'contain', transform: 'scaleX(-1)' }}
                                    />
                                    {!isStreaming && !error && (
                                        <Box sx={{ position: 'absolute', textAlign: 'center', opacity: 0.5 }}>
                                            <Camera size={64} style={{ marginBottom: 16 }} />
                                            <Typography>Camera is off</Typography>
                                        </Box>
                                    )}
                                </Paper>
                            </Grid>

                            {/* AI Analysis Feed */}
                            <Grid size={{ xs: 12, md: 6 }}>
                                <Typography variant="subtitle1" align="center" gutterBottom>AI Analysis</Typography>
                                <Paper
                                    elevation={0}
                                    sx={{
                                        p: 1,
                                        bgcolor: 'background.paper',
                                        borderRadius: 4,
                                        overflow: 'hidden',
                                        position: 'relative',
                                        aspectRatio: '3/4',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        border: '1px solid rgba(255,255,255,0.1)'
                                    }}
                                >
                                    <canvas
                                        ref={processedCanvasRef}
                                        width={480}
                                        height={640}
                                        style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                                    />
                                    {!isStreaming && (
                                        <Box sx={{ position: 'absolute', textAlign: 'center', opacity: 0.5 }}>
                                            <Typography>Waiting for stream...</Typography>
                                        </Box>
                                    )}
                                </Paper>
                            </Grid>
                        </Grid>
                    )}
                </Grid>

                <Grid size={{ xs: 12, lg: dualCameraMode ? 3 : 4 }}>
                    <Stack spacing={3}>
                        <Card sx={{ p: 3 }}>
                            <Typography variant="h6" gutterBottom>Session Stats</Typography>
                            <Stack spacing={2}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', p: 2, bgcolor: 'rgba(96, 165, 250, 0.1)', borderRadius: 2, border: '1px solid rgba(96, 165, 250, 0.2)' }}>
                                    <Typography color="text.secondary">Exercise</Typography>
                                    <Typography fontWeight="bold" variant="h6" color="primary.main">
                                        {currentExercise?.icon} {currentExercise?.label}
                                    </Typography>
                                </Box>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', p: 2, bgcolor: 'rgba(255,255,255,0.05)', borderRadius: 2 }}>
                                    <Typography color="text.secondary">Level</Typography>
                                    <Typography fontWeight="bold" variant="h6" sx={{ textTransform: 'capitalize' }}>
                                        {experienceLevel}
                                    </Typography>
                                </Box>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', p: 2, bgcolor: 'rgba(255,255,255,0.05)', borderRadius: 2 }}>
                                    <Typography color="text.secondary">Reps</Typography>
                                    <Typography fontWeight="bold" variant="h6">
                                        {currentReps}{targetReps ? ` / ${targetReps}` : ''}
                                    </Typography>
                                </Box>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', p: 2, bgcolor: 'rgba(255,255,255,0.05)', borderRadius: 2 }}>
                                    <Typography color="text.secondary">Form Score</Typography>
                                    <Typography fontWeight="bold" variant="h6" color="success.main">--</Typography>
                                </Box>
                                {dualCameraMode && (
                                    <Box sx={{
                                        p: 2,
                                        bgcolor: 'rgba(96, 165, 250, 0.05)',
                                        borderRadius: 2,
                                        border: '1px solid rgba(96, 165, 250, 0.2)'
                                    }}>
                                        <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
                                            Camera Status
                                        </Typography>
                                        <Stack direction="row" spacing={1}>
                                            <Chip
                                                size="small"
                                                label="Front"
                                                color={cameraStatus.front ? 'success' : 'default'}
                                                variant={cameraStatus.front ? 'filled' : 'outlined'}
                                            />
                                            <Chip
                                                size="small"
                                                label="Side"
                                                color={cameraStatus.side ? 'success' : 'default'}
                                                variant={cameraStatus.side ? 'filled' : 'outlined'}
                                            />
                                        </Stack>
                                    </Box>
                                )}
                            </Stack>
                        </Card>

                        <Card sx={{ p: 3 }}>
                            <Typography variant="h6" gutterBottom>
                                Instructions for {currentExercise?.label || 'Exercise'}
                            </Typography>
                            <Stack spacing={2}>
                                {dualCameraMode ? (
                                    /* Dual Camera Instructions */
                                    <>
                                        <Box sx={{ display: 'flex', gap: 2 }}>
                                            <CheckCircle2 size={20} color="#60a5fa" />
                                            <Typography variant="body2" color="text.secondary">
                                                Position front camera facing you
                                            </Typography>
                                        </Box>
                                        <Box sx={{ display: 'flex', gap: 2 }}>
                                            <CheckCircle2 size={20} color="#60a5fa" />
                                            <Typography variant="body2" color="text.secondary">
                                                Position side camera perpendicular (90°)
                                            </Typography>
                                        </Box>
                                        <Box sx={{ display: 'flex', gap: 2 }}>
                                            <CheckCircle2 size={20} color="#60a5fa" />
                                            <Typography variant="body2" color="text.secondary">
                                                Ensure full body visible in both views
                                            </Typography>
                                        </Box>
                                        {currentExercise?.dualCameraRecommended && (
                                            <Alert severity="success" sx={{ mt: 1 }}>
                                                <Typography variant="caption">
                                                    Dual camera recommended for {currentExercise.label}!
                                                </Typography>
                                            </Alert>
                                        )}
                                    </>
                                ) : (
                                    /* Single Camera Instructions */
                                    <>
                                        <Box sx={{ display: 'flex', gap: 2 }}>
                                            <CheckCircle2 size={20} color="#60a5fa" />
                                            <Typography variant="body2" color="text.secondary">Ensure your full body is visible</Typography>
                                        </Box>
                                        <Box sx={{ display: 'flex', gap: 2 }}>
                                            <CheckCircle2 size={20} color="#60a5fa" />
                                            <Typography variant="body2" color="text.secondary">
                                                {(exerciseType === 'pullup' || exerciseType === 'dips')
                                                    ? 'Position yourself facing the camera'
                                                    : (exerciseType === 'plank' || exerciseType === 'wall_sit' || exerciseType === 'glute_bridge')
                                                        ? 'Position yourself sideways to the camera'
                                                        : 'Face the camera initially, then turn to the side'}
                                            </Typography>
                                        </Box>
                                        <Box sx={{ display: 'flex', gap: 2 }}>
                                            <CheckCircle2 size={20} color="#60a5fa" />
                                            <Typography variant="body2" color="text.secondary">
                                                {exerciseType === 'squat' && 'Maintain proper form throughout the movement'}
                                                {exerciseType === 'pullup' && 'Start from a dead hang with arms extended'}
                                                {exerciseType === 'pushup' && 'Keep your body in a straight line'}
                                                {exerciseType === 'dips' && 'Lower until your elbows are at 90 degrees'}
                                                {exerciseType === 'lunges' && 'Step forward with controlled movement'}
                                                {exerciseType === 'plank' && 'Keep your body in a straight line from head to heels'}
                                                {exerciseType === 'deadlift' && 'Keep your back straight and hinge at the hips'}
                                                {exerciseType === 'overhead_press' && 'Press the weight directly overhead with full lockout'}
                                                {exerciseType === 'bent_over_row' && 'Maintain hip hinge and neutral spine while rowing'}
                                                {exerciseType === 'glute_bridge' && 'Drive through your heels and squeeze glutes at the top'}
                                                {exerciseType === 'wall_sit' && 'Keep your back flat against the wall, knees at 90°'}
                                            </Typography>
                                        </Box>
                                        {currentExercise?.dualCameraRecommended && (
                                            <Alert severity="info" sx={{ mt: 1 }}>
                                                <Typography variant="caption">
                                                    💡 Enable Dual Camera for better analysis of {currentExercise.label}
                                                </Typography>
                                            </Alert>
                                        )}
                                    </>
                                )}
                            </Stack>
                        </Card>
                    </Stack>
                </Grid>
            </Grid>
        </Box>
    );
};
