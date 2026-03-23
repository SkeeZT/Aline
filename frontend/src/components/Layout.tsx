import React from 'react';
import { Link as RouterLink, useLocation } from 'react-router-dom';
import { AppBar, Toolbar, Container, Button, Box, Typography } from '@mui/material';
import { Activity, Video, Upload } from 'lucide-react';

interface LayoutProps {
    children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
    const location = useLocation();

    const isActive = (path: string) => location.pathname === path;

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
            <AppBar position="sticky">
                <Container maxWidth="lg">
                    <Toolbar disableGutters>
                        <Activity style={{ marginRight: 8, color: '#60a5fa' }} />
                        <Typography
                            variant="h6"
                            noWrap
                            component={RouterLink}
                            to="/"
                            sx={{
                                mr: 4,
                                fontWeight: 700,
                                color: 'inherit',
                                textDecoration: 'none',
                                background: 'linear-gradient(135deg, #60a5fa, #a78bfa)',
                                WebkitBackgroundClip: 'text',
                                WebkitTextFillColor: 'transparent',
                                flexGrow: { xs: 1, md: 0 }
                            }}
                        >
                            AI Trainer
                        </Typography>

                        <Box sx={{ flexGrow: 1, display: 'flex', gap: 2 }}>
                            <Button
                                component={RouterLink}
                                to="/live"
                                startIcon={<Video size={18} />}
                                color={isActive('/live') ? 'primary' : 'inherit'}
                                sx={{ opacity: isActive('/live') ? 1 : 0.7 }}
                            >
                                Live Analysis
                            </Button>
                            <Button
                                component={RouterLink}
                                to="/upload"
                                startIcon={<Upload size={18} />}
                                color={isActive('/upload') ? 'primary' : 'inherit'}
                                sx={{ opacity: isActive('/upload') ? 1 : 0.7 }}
                            >
                                Upload Video
                            </Button>
                        </Box>
                    </Toolbar>
                </Container>
            </AppBar>

            <Container component="main" maxWidth="lg" sx={{ flexGrow: 1, py: 4 }}>
                {children}
            </Container>
        </Box>
    );
};
