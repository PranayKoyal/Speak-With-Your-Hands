import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AppProvider } from './context/AppContext';
import { SettingsProvider } from './context/SettingsContext';
import { RecognitionProvider } from './context/RecognitionContext';
import Home from './pages/Home';
import Recognition from './pages/Recognition';
import Settings from './pages/Settings';
import About from './pages/About';

function App() {
  return (
    <Router>
      <AppProvider>
        <SettingsProvider>
          <RecognitionProvider>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/recognition" element={<Recognition />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="/about" element={<About />} />
            </Routes>
          </RecognitionProvider>
        </SettingsProvider>
      </AppProvider>
    </Router>
  );
}

export default App;
