import { createTheme, alpha } from '@mui/material/styles';

const theme = createTheme({
    palette: {
        mode: 'dark',
        primary: {
            main: '#3b82f6', // Blue 500
            light: '#60a5fa',
            dark: '#2563eb',
        },
        secondary: {
            main: '#8b5cf6', // Violet 500
            light: '#a78bfa',
            dark: '#7c3aed',
        },
        background: {
            default: '#0f172a', // Slate 900
            paper: '#1e293b', // Slate 800
        },
        text: {
            primary: '#f8fafc', // Slate 50
            secondary: '#94a3b8', // Slate 400
        },
    },
    typography: {
        fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
        h1: {
            fontWeight: 700,
        },
        h2: {
            fontWeight: 700,
        },
        h3: {
            fontWeight: 600,
        },
        button: {
            textTransform: 'none',
            fontWeight: 600,
        },
    },
    components: {
        MuiCssBaseline: {
            styleOverrides: {
                body: {
                    backgroundImage: `
            radial-gradient(at 0% 0%, ${alpha('#3b82f6', 0.15)} 0px, transparent 50%),
            radial-gradient(at 100% 0%, ${alpha('#8b5cf6', 0.15)} 0px, transparent 50%)
          `,
                    backgroundAttachment: 'fixed',
                },
            },
        },
        MuiCard: {
            styleOverrides: {
                root: {
                    backgroundColor: alpha('#1e293b', 0.7),
                    backdropFilter: 'blur(12px)',
                    border: `1px solid ${alpha('#ffffff', 0.1)}`,
                    borderRadius: '16px',
                },
            },
        },
        MuiPaper: {
            styleOverrides: {
                root: {
                    backgroundImage: 'none',
                },
            },
        },
        MuiButton: {
            styleOverrides: {
                root: {
                    borderRadius: '8px',
                    padding: '10px 24px',
                },
                containedPrimary: {
                    background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
                    '&:hover': {
                        background: 'linear-gradient(135deg, #2563eb, #7c3aed)',
                    },
                },
            },
        },
        MuiAppBar: {
            styleOverrides: {
                root: {
                    backgroundColor: alpha('#1e293b', 0.7),
                    backdropFilter: 'blur(12px)',
                    borderBottom: `1px solid ${alpha('#ffffff', 0.1)}`,
                    boxShadow: 'none',
                },
            },
        },
    },
});

export default theme;
