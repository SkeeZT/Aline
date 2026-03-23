import React, { useEffect, useState, useRef } from 'react';
import { Box, Typography, Card, Grid, Button, Stack, Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Chip } from '@mui/material';
import { CheckCircle2, XCircle, ArrowLeft, Activity, Zap, BarChart2 } from 'lucide-react';
import { useLocation, useNavigate } from 'react-router-dom';
import { apiRequest, API_BASE_URL } from '../utils/api';

interface AnalysisResultsState {
    videoPath: string;
    filename: string;
}

interface AnalysisDetails {
    summary: {
        total_reps: number;
        successful_reps: number;
        unsuccessful_reps: number;
    };
    reps_detail: Array<{
        rep_number: number;
        success: boolean;
        issues: string[];
        failure_justifications?: Record<string, string>;
    }>;
}

interface VBTAnalysis {
    summary_statistics: {
        concentric_velocity_stats: { mean: number; max: number };
        peak_velocity_stats: { mean: number; max: number };
        quality_stats: { mean: number };
    };
    rep_velocities: Array<{
        rep_number: number;
        concentric_velocity: number;
        peak_velocity: number;
        total_duration: number;
        rep_quality_score: number;
    }>;
}

export const AnalysisResults: React.FC = () => {
    const location = useLocation();
    const navigate = useNavigate();

    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const state = location.state as AnalysisResultsState;

    const [status, setStatus] = useState<'processing' | 'completed' | 'error'>('processing');
    const [videoUrl, setVideoUrl] = useState<string | null>(null);
    const [videoLoading, setVideoLoading] = useState(false);
    const [firstFrameReceived, setFirstFrameReceived] = useState(false);

    // Data States
    const [analysisData, setAnalysisData] = useState<AnalysisDetails | null>(null);
    const [vbtData, setVbtData] = useState<VBTAnalysis | null>(null);

    useEffect(() => {
        if (!state?.filename) {
            navigate('/upload');
            return;
        }

        let ws: WebSocket | null = null;
        if (status === 'processing') {
            const videoId = state.filename.split('.')[0];
            const wsUrl = API_BASE_URL.replace('http', 'ws') + `/api/video/monitor/${videoId}`;
            try {
                ws = new WebSocket(wsUrl);
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.frame && canvasRef.current) {
                        setFirstFrameReceived(true);
                        const ctx = canvasRef.current.getContext('2d');
                        if (ctx) {
                            const img = new Image();
                            img.onload = () => {
                                ctx.drawImage(img, 0, 0, canvasRef.current!.width, canvasRef.current!.height);
                            };
                            img.src = data.frame;
                        }
                    }
                };
            } catch (e) {
                console.error("WebSocket error:", e);
            }
        }

        const checkStatus = async () => {
            try {
                const response = await apiRequest(`/api/video/status/${state.filename}`);
                if (response.ok) {
                    const data = await response.json();
                    if (data.status === 'completed') {
                        setVideoUrl(data.video_url);
                        setStatus('completed');
                        setVideoLoading(true);
                        if (ws) ws.close();

                        // Fetch detailed results
                        if (data.details_url) {
                            fetch(data.details_url)
                                .then(res => res.json())
                                .then(setAnalysisData)
                                .catch(err => console.error("Error fetching analysis details:", err));
                        }
                        if (data.vbt_url) {
                            fetch(data.vbt_url)
                                .then(res => res.json())
                                .then(setVbtData)
                                .catch(err => console.error("Error fetching VBT data:", err));
                        }
                    }
                }
            } catch (error) {
                console.error('Error checking status:', error);
            }
        };

        const interval = status === 'processing' ? setInterval(checkStatus, 2000) : null;
        if (status === 'processing') {
            checkStatus();
        }

        return () => {
            if (interval) clearInterval(interval);
            if (ws) ws.close();
        };
    }, [state, navigate, status]);


    if (!state) return null;

    if (status === 'processing') {
        return (
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '60vh' }}>
                <Typography variant="h5" gutterBottom>Analyzing Video...</Typography>
                <Typography color="text.secondary" sx={{ mb: 4 }}>Live preview of analysis:</Typography>
                <Paper
                    elevation={3}
                    sx={{
                        p: 1,
                        bgcolor: 'black',
                        borderRadius: 2,
                        border: '1px solid rgba(255,255,255,0.1)',
                        width: '100%',
                        maxWidth: '640px',
                        aspectRatio: '9/16',
                        position: 'relative',
                    }}
                >
                    {!firstFrameReceived && (
                        <Box sx={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', bgcolor: 'rgba(0, 0, 0, 0.8)', zIndex: 2, borderRadius: 2 }}>
                            <Box sx={{ width: 50, height: 50, border: '4px solid rgba(255, 255, 255, 0.3)', borderTop: '4px solid #60a5fa', borderRadius: '50%', animation: 'spin 1s linear infinite', mb: 2 }} />
                            <Typography variant="body2" color="white">Waiting for analysis stream...</Typography>
                        </Box>
                    )}
                    <canvas ref={canvasRef} height={1280} width={720} style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                </Paper>
                <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
            </Box>
        );
    }

    const totalReps = analysisData?.summary.total_reps || 0;
    const successfulReps = analysisData?.summary.successful_reps || 0;
    const formScore = totalReps > 0 ? Math.round((successfulReps / totalReps) * 100) : 0;

    // Combine reps data
    const combinedReps = analysisData?.reps_detail.map(rep => {
        const vbt = vbtData?.rep_velocities.find(v => v.rep_number === rep.rep_number);
        return { ...rep, ...vbt };
    }) || [];

    return (
        <Box sx={{ maxWidth: 'xl', mx: 'auto', pb: 8, px: 2 }}>
            <Button startIcon={<ArrowLeft />} onClick={() => navigate('/upload')} sx={{ mb: 4 }}>
                Back to Upload
            </Button>

            <Grid container spacing={4}>
                {/* Left Column: Video */}
                <Grid size={{ xs: 12, lg: 7 }}>
                    <Paper elevation={0} sx={{ overflow: 'hidden', borderRadius: 4, bgcolor: 'black', aspectRatio: '16/9', position: 'relative', border: '1px solid rgba(255,255,255,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        {videoLoading && (
                            <Box sx={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', bgcolor: 'rgba(0, 0, 0, 0.8)', zIndex: 2 }}>
                                <Box sx={{ width: 50, height: 50, border: '4px solid rgba(255, 255, 255, 0.3)', borderTop: '4px solid #60a5fa', borderRadius: '50%', animation: 'spin 1s linear infinite', mb: 2 }} />
                                <Typography variant="body2" color="white">Loading video...</Typography>
                            </Box>
                        )}
                        <video
                            ref={videoRef}
                            src={videoUrl || ''}
                            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                            onLoadedData={() => {
                                console.log("Video loaded:", videoUrl);
                                setVideoLoading(false);
                            }}
                            onCanPlay={() => setVideoLoading(false)}
                            onError={(e) => {
                                console.error("Video error:", e);
                                console.error("Video URL was:", videoUrl);
                                setVideoLoading(false);
                            }}
                            controls
                        />
                    </Paper>

                    {/* VBT Stats Section */}
                    {vbtData && (
                        <Grid container spacing={2} sx={{ mt: 2 }}>
                            <Grid size={{ xs: 12, md: 4 }}>
                                <Card sx={{ p: 2, bgcolor: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.2)' }}>
                                    <Stack direction="row" spacing={1} alignItems="center" mb={1}>
                                        <Zap size={18} color="#60a5fa" />
                                        <Typography variant="body2" color="text.secondary">Avg Velocity</Typography>
                                    </Stack>
                                    <Typography variant="h5" fontWeight="bold">
                                        {vbtData.summary_statistics.concentric_velocity_stats.mean.toFixed(2)} m/s
                                    </Typography>
                                </Card>
                            </Grid>
                            <Grid size={{ xs: 12, md: 4 }}>
                                <Card sx={{ p: 2, bgcolor: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
                                    <Stack direction="row" spacing={1} alignItems="center" mb={1}>
                                        <Activity size={18} color="#34d399" />
                                        <Typography variant="body2" color="text.secondary">Peak Velocity</Typography>
                                    </Stack>
                                    <Typography variant="h5" fontWeight="bold">
                                        {vbtData.summary_statistics.peak_velocity_stats.max.toFixed(2)} m/s
                                    </Typography>
                                </Card>
                            </Grid>
                            <Grid size={{ xs: 12, md: 4 }}>
                                <Card sx={{ p: 2, bgcolor: 'rgba(236, 72, 153, 0.1)', border: '1px solid rgba(236, 72, 153, 0.2)' }}>
                                    <Stack direction="row" spacing={1} alignItems="center" mb={1}>
                                        <BarChart2 size={18} color="#f472b6" />
                                        <Typography variant="body2" color="text.secondary">Quality Score</Typography>
                                    </Stack>
                                    <Typography variant="h5" fontWeight="bold">
                                        {(vbtData.summary_statistics.quality_stats.mean * 100).toFixed(0)}%
                                    </Typography>
                                </Card>
                            </Grid>
                        </Grid>
                    )}
                </Grid>

                {/* Right Column: Stats & Reps */}
                <Grid size={{ xs: 12, lg: 5 }}>
                    <Stack spacing={3}>
                        <Card sx={{ p: 3 }}>
                            <Typography variant="h6" gutterBottom>Analysis Summary</Typography>
                            <Stack spacing={2}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', p: 2, bgcolor: 'rgba(255,255,255,0.05)', borderRadius: 2 }}>
                                    <Typography color="text.secondary">Total Reps</Typography>
                                    <Typography fontWeight="bold" variant="h6">{totalReps}</Typography>
                                </Box>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', p: 2, bgcolor: 'rgba(255,255,255,0.05)', borderRadius: 2 }}>
                                    <Typography color="text.secondary">Form Accuracy</Typography>
                                    <Typography fontWeight="bold" variant="h6" color={formScore > 80 ? "success.main" : "warning.main"}>
                                        {formScore}%
                                    </Typography>
                                </Box>
                            </Stack>
                        </Card>

                        {/* Detailed Rep Log */}
                        <Card sx={{ overflow: 'hidden' }}>
                            <Box sx={{ p: 2, borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                                <Typography variant="h6">Rep Breakdown</Typography>
                            </Box>
                            <TableContainer sx={{ maxHeight: 400 }}>
                                <Table stickyHeader size="small">
                                    <TableHead>
                                        <TableRow>
                                            <TableCell>Rep</TableCell>
                                            <TableCell>Status</TableCell>
                                            <TableCell>Velocity</TableCell>
                                            <TableCell>Issues</TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {combinedReps.map((rep) => (
                                            <TableRow key={rep.rep_number}>
                                                <TableCell>#{rep.rep_number}</TableCell>
                                                <TableCell>
                                                    {rep.success ?
                                                        <Chip label="Good" color="success" size="small" icon={<CheckCircle2 size={12} />} /> :
                                                        <Chip label="Bad" color="error" size="small" icon={<XCircle size={12} />} />
                                                    }
                                                </TableCell>
                                                <TableCell sx={{ fontFamily: 'monospace' }}>
                                                    {rep.concentric_velocity ? `${rep.concentric_velocity.toFixed(2)}m/s` : '-'}
                                                </TableCell>
                                                <TableCell>
                                                    {rep.issues && rep.issues.length > 0 ? (
                                                        <Stack spacing={0.5}>
                                                            {rep.issues.map((issue, i) => (
                                                                <Typography key={i} variant="caption" color="error.light" sx={{ display: 'block' }}>
                                                                    • {issue.replace(/_/g, ' ')}
                                                                </Typography>
                                                            ))}
                                                        </Stack>
                                                    ) : (
                                                        <Typography variant="caption" color="text.secondary">-</Typography>
                                                    )}
                                                    {rep.failure_justifications && Object.values(rep.failure_justifications).map((just, i) => (
                                                        <Typography key={`just-${i}`} variant="caption" color="text.secondary" display="block">
                                                            ({just})
                                                        </Typography>
                                                    ))}
                                                </TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </TableContainer>
                        </Card>
                    </Stack>
                </Grid>
            </Grid>
            <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
        </Box>
    );
};
