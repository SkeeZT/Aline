import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Box, Typography, Grid, Card, CardContent, Button, useTheme } from '@mui/material';
import { Video, Upload, ArrowRight } from 'lucide-react';

export const Home: React.FC = () => {
    const navigate = useNavigate();
    const theme = useTheme();

    return (
        <Box sx={{ maxWidth: 'md', mx: 'auto', mt: 4 }}>
            <Box sx={{ textAlign: 'center', mb: 8 }}>
                <Typography variant="h2" component="h1" gutterBottom sx={{ fontWeight: 800 }}>
                    Master Your Form with{' '}
                    <Box component="span" sx={{
                        background: 'linear-gradient(135deg, #60a5fa, #a78bfa)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                    }}>
                        AI Precision
                    </Box>
                </Typography>
                <Typography variant="h5" color="text.secondary" sx={{ maxWidth: 'sm', mx: 'auto' }}>
                    Real-time pose estimation and form correction for 8 exercises:
                    Squats, Deadlifts, Pull-ups, Push-ups, Dips, Lunges, Planks, and Bench Press.
                </Typography>
            </Box>

            <Grid container spacing={4}>
                <Grid size={{ xs: 12, md: 6 }}>
                    <Card
                        sx={{
                            height: '100%',
                            cursor: 'pointer',
                            transition: 'all 0.3s ease',
                            '&:hover': {
                                borderColor: theme.palette.primary.main,
                                transform: 'translateY(-4px)',
                            }
                        }}
                        onClick={() => navigate('/live')}
                    >
                        <CardContent sx={{ p: 4, height: '100%', display: 'flex', flexDirection: 'column', gap: 3 }}>
                            <Box sx={{
                                width: 48,
                                height: 48,
                                borderRadius: 2,
                                bgcolor: 'rgba(59, 130, 246, 0.1)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                color: 'primary.main'
                            }}>
                                <Video size={24} />
                            </Box>

                            <Box>
                                <Typography variant="h5" gutterBottom fontWeight="bold">
                                    Live Analysis
                                </Typography>
                                <Typography variant="body1" color="text.secondary">
                                    Use your webcam for real-time form feedback.
                                    Perfect for home workouts and instant correction.
                                </Typography>
                            </Box>

                            <Button
                                variant="contained"
                                fullWidth
                                endIcon={<ArrowRight size={16} />}
                                sx={{ mt: 'auto' }}
                            >
                                Start Session
                            </Button>
                        </CardContent>
                    </Card>
                </Grid>

                <Grid size={{ xs: 12, md: 6 }}>
                    <Card
                        sx={{
                            height: '100%',
                            cursor: 'pointer',
                            transition: 'all 0.3s ease',
                            '&:hover': {
                                borderColor: theme.palette.secondary.main,
                                transform: 'translateY(-4px)',
                            }
                        }}
                        onClick={() => navigate('/upload')}
                    >
                        <CardContent sx={{ p: 4, height: '100%', display: 'flex', flexDirection: 'column', gap: 3 }}>
                            <Box sx={{
                                width: 48,
                                height: 48,
                                borderRadius: 2,
                                bgcolor: 'rgba(139, 92, 246, 0.1)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                color: 'secondary.main'
                            }}>
                                <Upload size={24} />
                            </Box>

                            <Box>
                                <Typography variant="h5" gutterBottom fontWeight="bold">
                                    Upload Video
                                </Typography>
                                <Typography variant="body1" color="text.secondary">
                                    Analyze pre-recorded videos for detailed breakdown.
                                    Great for reviewing past sessions.
                                </Typography>
                            </Box>

                            <Button
                                variant="outlined"
                                color="secondary"
                                fullWidth
                                endIcon={<ArrowRight size={16} />}
                                sx={{ mt: 'auto' }}
                            >
                                Upload File
                            </Button>
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>
        </Box>
    );
};
