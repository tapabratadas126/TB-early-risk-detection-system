import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { LanguageProvider } from './context/LanguageContext';
import HomePage from './pages/HomePage';
import AssessmentPage from './pages/AssessmentPage';
import ResultsPage from './pages/ResultsPage';
import LearnPage from './pages/LearnPage';
import PreventionPage from './pages/PreventionPage';
import FindHelpPage from './pages/FindHelpPage';
import PrivacyPage from './pages/PrivacyPage';

export default function App() {
  return (
    <LanguageProvider>
      <Router>
        <Routes>
          <Route path="/" element={<HomePage />} /> 
          <Route path="/assessment" element={<AssessmentPage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/learn" element={<LearnPage />} />
          <Route path="/prevention" element={<PreventionPage />} />
          <Route path="/find-help" element={<FindHelpPage />} />
          <Route path="/privacy" element={<PrivacyPage />} />
        </Routes>
      </Router>
    </LanguageProvider>
  );
}