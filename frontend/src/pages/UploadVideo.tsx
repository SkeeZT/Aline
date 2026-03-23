import React, { useState, useRef, useEffect } from 'react';
import { Box, Typography, Button, Card, LinearProgress, Alert, Paper, Stack, FormControl, InputLabel, Select, MenuItem } from '@mui/material';
import type { SelectChangeEvent } from '@mui/material';
import { Upload, FileVideo, CheckCircle, XCircle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { apiRequest } from '../utils/api';

// Exercise options (shared with LiveAnalysis)
const EXERCISES = [
    { value: 'squat', label: 'Squat', icon: '🏋️' },
    { value: 'pullup', label: 'Pull-up', icon: '💪' },
    { value: 'pushup', label: 'Push-up', icon: '🙌' },
    { value: 'dips', label: 'Dips', icon: '⬇️' },
    { value: 'lunges', label: 'Lunges', icon: '🦵' },
    { value: 'plank', label: 'Plank', icon: '🧘' },
    { value: 'deadlift', label: 'Deadlift', icon: '🏋️‍♂️' },
    { value: 'overhead_press', label: 'Overhead Press', icon: '🙆' },
    { value: 'bent_over_row', label: 'Bent-Over Row', icon: '🚣' },
    { value: 'glute_bridge', label: 'Glute Bridge', icon: '🌉' },
    { value: 'wall_sit', label: 'Wall Sit', icon: '🧱' },
    { value: 'bench_press', label: 'Bench Press', icon: '🏋️‍♂️' },
];

const EXPERIENCE_LEVELS = [
    { value: 'beginner', label: 'Beginner' },
    { value: 'intermediate', label: 'Intermediate' },
    { value: 'advanced', label: 'Advanced' },
];

export const UploadVideo: React.FC = () => {
    const navigate = useNavigate();
    const [file, setFile] = useState<File | null>(null);
    const [uploading, setUploading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [success, setSuccess] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null);
    const [videoLoading, setVideoLoading] = useState(false);
    const [exerciseType, setExerciseType] = useState<string>('squat');
    const [experienceLevel, setExperienceLevel] = useState<string>('intermediate');
    const fileInputRef = useRef<HTMLInputElement>(null);
    const videoRef = useRef<HTMLVideoElement>(null);

    // Cleanup video preview URL on unmount
    useEffect(() => {
        return () => {
            if (videoPreviewUrl) {
                URL.revokeObjectURL(videoPreviewUrl);
            }
        };
    }, [videoPreviewUrl]);

    const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files[0]) {
            const selectedFile = event.target.files[0];

            // Clean up previous preview URL if exists
            if (videoPreviewUrl) {
                URL.revokeObjectURL(videoPreviewUrl);
            }

            setFile(selectedFile);
            setError(null);
            setSuccess(false);
            setProgress(0);

            // Create preview URL for video
            setVideoLoading(true);
            const url = URL.createObjectURL(selectedFile);
            setVideoPreviewUrl(url);
        }
    };

    const handleUpload = async () => {
        if (!file) return;

        setUploading(true);
        setProgress(0);
        setError(null);

        const formData = new FormData();
        formData.append('file', file);

        try {
            // Upload the file using fetch instead of XMLHttpRequest
            const uploadResponse = await apiRequest('/api/video/upload', {
                method: 'POST',
                body: formData,
            });

            if (uploadResponse.ok) {
                const response = await uploadResponse.json();

                // Start analysis with exercise type and experience level
                try {
                    const analyzeParams = new URLSearchParams({
                        video_path: response.path,
                        exercise_type: exerciseType,
                        experience_level: experienceLevel
                    });
                    const analyzeResponse = await apiRequest(`/api/video/analyze?${analyzeParams.toString()}`, {
                        method: 'POST'
                    });

                    if (analyzeResponse.ok) {
                        setSuccess(true);
                        // Navigate to results page with video info
                        navigate('/results', {
                            state: {
                                videoPath: response.path,
                                filename: response.filename
                            }
                        });
                    } else {
                        setError('Upload successful, but analysis failed to start.');
                    }
                } catch (err) {
                    console.error(err);
                    setError('Error starting analysis.');
                }
            } else {
                setError('Upload failed. Please try again.');
            }
            setUploading(false);
        } catch (err) {
            console.error(err);
            setError('An unexpected error occurred.');
            setUploading(false);
        }
    };

    return (
        <Box sx={{ maxWidth: 'md', mx: 'auto' }}>
            <Typography variant="h4" fontWeight="bold" gutterBottom>
                Upload Video
            </Typography>
            <Typography color="text.secondary" sx={{ mb: 3 }}>
                Upload a video of your exercise for detailed AI-powered form analysis.
            </Typography>

            {/* Exercise Selection */}
            <Box sx={{ display: 'flex', gap: 2, mb: 4, flexWrap: 'wrap' }}>
                <FormControl size="small" sx={{ minWidth: 160 }}>
                    <InputLabel id="upload-exercise-label">Exercise Type</InputLabel>
                    <Select
                        labelId="upload-exercise-label"
                        value={exerciseType}
                        label="Exercise Type"
                        onChange={(e: SelectChangeEvent) => setExerciseType(e.target.value)}
                        disabled={uploading}
                    >
                        {EXERCISES.map((ex) => (
                            <MenuItem key={ex.value} value={ex.value}>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                    <span>{ex.icon}</span>
                                    <span>{ex.label}</span>
                                </Box>
                            </MenuItem>
                        ))}
                    </Select>
                </FormControl>

                <FormControl size="small" sx={{ minWidth: 140 }}>
                    <InputLabel id="upload-level-label">Experience Level</InputLabel>
                    <Select
                        labelId="upload-level-label"
                        value={experienceLevel}
                        label="Experience Level"
                        onChange={(e: SelectChangeEvent) => setExperienceLevel(e.target.value)}
                        disabled={uploading}
                    >
                        {EXPERIENCE_LEVELS.map((level) => (
                            <MenuItem key={level.value} value={level.value}>
                                {level.label}
                            </MenuItem>
                        ))}
                    </Select>
                </FormControl>
            </Box>

            <Card sx={{ p: 6, textAlign: 'center' }}>
                <input
                    type="file"
                    accept="video/*"
                    style={{ display: 'none' }}
                    ref={fileInputRef}
                    onChange={handleFileSelect}
                />

                {!file && !success && (
                    <Box
                        sx={{
                            border: '2px dashed rgba(255,255,255,0.1)',
                            borderRadius: 4,
                            p: 8,
                            cursor: 'pointer',
                            transition: 'all 0.2s',
                            '&:hover': {
                                borderColor: 'primary.main',
                                bgcolor: 'rgba(59, 130, 246, 0.05)'
                            }
                        }}
                        onClick={() => fileInputRef.current?.click()}
                    >
                        <Upload size={48} style={{ marginBottom: 16, color: '#60a5fa' }} />
                        <Typography variant="h6" gutterBottom>
                            Click to select video
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                            MP4, MOV, or AVI files supported
                        </Typography>
                    </Box>
                )}

                {file && (
                    <Stack spacing={4} alignItems="center">
                        {/* Video Preview */}
                        {videoPreviewUrl && (
                            <Paper
                                elevation={0}
                                sx={{
                                    width: '100%',
                                    position: 'relative',
                                    bgcolor: 'black',
                                    borderRadius: 4,
                                    overflow: 'hidden',
                                    aspectRatio: '16/9',
                                    border: '1px solid rgba(255,255,255,0.1)',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center'
                                }}
                            >
                                {videoLoading && (
                                    <Box
                                        sx={{
                                            position: 'absolute',
                                            top: 0,
                                            left: 0,
                                            right: 0,
                                            bottom: 0,
                                            display: 'flex',
                                            flexDirection: 'column',
                                            alignItems: 'center',
                                            justifyContent: 'center',
                                            bgcolor: 'rgba(0, 0, 0, 0.8)',
                                            zIndex: 2
                                        }}
                                    >
                                        <Box
                                            sx={{
                                                width: 50,
                                                height: 50,
                                                border: '4px solid rgba(255, 255, 255, 0.3)',
                                                borderTop: '4px solid #60a5fa',
                                                borderRadius: '50%',
                                                animation: 'spin 1s linear infinite',
                                                mb: 2
                                            }}
                                        />
                                        <Typography variant="body2" color="white">
                                            Loading preview...
                                        </Typography>
                                    </Box>
                                )}
                                <video
                                    ref={videoRef}
                                    src={videoPreviewUrl}
                                    style={{
                                        width: '100%',
                                        height: '100%',
                                        objectFit: 'contain'
                                    }}
                                    onLoadedData={() => {
                                        setVideoLoading(false);
                                    }}
                                    onCanPlay={() => {
                                        setVideoLoading(false);
                                    }}
                                    controls
                                />
                            </Paper>
                        )}

                        <Paper sx={{ p: 3, width: '100%', display: 'flex', alignItems: 'center', gap: 2, bgcolor: 'rgba(255,255,255,0.05)' }}>
                            <FileVideo size={32} color="#60a5fa" />
                            <Box sx={{ flexGrow: 1, textAlign: 'left' }}>
                                <Typography variant="subtitle1" fontWeight="medium">
                                    {file.name}
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                    {(file.size / (1024 * 1024)).toFixed(2)} MB
                                </Typography>
                            </Box>
                            <Button
                                color="error"
                                onClick={() => {
                                    setFile(null);
                                    if (videoPreviewUrl) {
                                        URL.revokeObjectURL(videoPreviewUrl);
                                        setVideoPreviewUrl(null);
                                    }
                                    setVideoLoading(false);
                                }}
                                disabled={uploading}
                            >
                                <XCircle />
                            </Button>
                        </Paper>

                        {uploading && (
                            <Box sx={{ width: '100%' }}>
                                <LinearProgress variant="determinate" value={progress} sx={{ height: 8, borderRadius: 4 }} />
                                <Typography variant="body2" color="text.secondary" sx={{ mt: 1, textAlign: 'right' }}>
                                    {Math.round(progress)}%
                                </Typography>
                            </Box>
                        )}

                        <Button
                            variant="contained"
                            size="large"
                            onClick={handleUpload}
                            disabled={uploading}
                            fullWidth
                        >
                            {uploading ? 'Uploading...' : 'Start Analysis'}
                        </Button>
                    </Stack>
                )}

                {success && (
                    <Box sx={{ textAlign: 'center', py: 4 }}>
                        <CheckCircle size={64} color="#4ade80" style={{ marginBottom: 16 }} />
                        <Typography variant="h5" gutterBottom>
                            Upload Successful!
                        </Typography>
                        <Typography color="text.secondary" sx={{ mb: 4 }}>
                            Your video is being analyzed. You will be notified when results are ready.
                        </Typography>
                        <Button variant="outlined" onClick={() => setSuccess(false)}>
                            Upload Another
                        </Button>
                    </Box>
                )}

                {error && (
                    <Alert severity="error" sx={{ mt: 3 }}>
                        {error}
                    </Alert>
                )}
            </Card>

            <style>{`
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            `}</style>
        </Box>
    );
};
