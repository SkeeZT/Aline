import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, CssBaseline } from '@mui/material';
import theme from './theme';
import { Layout } from './components/Layout';
import { Home } from './pages/Home';
import { LiveAnalysis } from './pages/LiveAnalysis';
import { UploadVideo } from './pages/UploadVideo';
import { AnalysisResults } from './pages/AnalysisResults';

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/live" element={<LiveAnalysis />} />
            <Route path="/upload" element={<UploadVideo />} />
            <Route path="/results" element={<AnalysisResults />} />
          </Routes>
        </Layout>
      </Router>
    </ThemeProvider>
  );
}

export default App;
