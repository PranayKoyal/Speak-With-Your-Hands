import React from 'react';
import Layout from '../components/layout/Layout';
import { useSettings } from '../context/SettingsContext';
import { api } from '../services/api';

export const Settings: React.FC = () => {
  const { settings, updateSettings } = useSettings();

  const handleSettingChange = async (key: string, value: number | boolean) => {
    const newSettings = { ...settings, [key]: value };
    updateSettings(newSettings);
    
    try {
      await api.updateSettings({
        modelSensitivity: newSettings.modelSensitivity,
        gestureCooldown: newSettings.gestureCooldown,
        showLandmarks: newSettings.showLandmarks,
      });
    } catch (error) {
      console.error('Failed to update settings:', error);
    }
  };

  return (
    <Layout>
      <div className="max-w-3xl mx-auto">
        <h1 className="text-2xl font-bold mb-6">Settings</h1>
        
        <div className="bg-white p-6 rounded-lg shadow-md mb-6">
          <h2 className="text-xl font-semibold mb-4">Model Settings</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Model Sensitivity
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={settings.modelSensitivity}
                onChange={(e) => handleSettingChange('modelSensitivity', parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="text-sm text-gray-500 mt-1">
                {settings.modelSensitivity.toFixed(1)}
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Gesture Cooldown (ms)
              </label>
              <input
                type="range"
                min="500"
                max="2000"
                step="100"
                value={settings.gestureCooldown}
                onChange={(e) => handleSettingChange('gestureCooldown', parseInt(e.target.value))}
                className="w-full"
              />
              <div className="text-sm text-gray-500 mt-1">
                {settings.gestureCooldown}ms
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4">UI Settings</h2>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-gray-700">
                Show Landmarks
              </label>
              <div className="relative inline-block w-12 h-6">
                <input
                  type="checkbox"
                  checked={settings.showLandmarks}
                  onChange={(e) => handleSettingChange('showLandmarks', e.target.checked)}
                  className="sr-only"
                />
                <div
                  className={`block w-12 h-6 rounded-full transition-colors ${
                    settings.showLandmarks ? 'bg-blue-600' : 'bg-gray-300'
                  }`}
                />
                <div
                  className={`absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition-transform transform ${
                    settings.showLandmarks ? 'translate-x-6' : ''
                  }`}
                />
              </div>
            </div>

            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-gray-700">
                Enable Voice
              </label>
              <div className="relative inline-block w-12 h-6">
                <input
                  type="checkbox"
                  checked={settings.enableVoice}
                  onChange={(e) => handleSettingChange('enableVoice', e.target.checked)}
                  className="sr-only"
                />
                <div
                  className={`block w-12 h-6 rounded-full transition-colors ${
                    settings.enableVoice ? 'bg-blue-600' : 'bg-gray-300'
                  }`}
                />
                <div
                  className={`absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition-transform transform ${
                    settings.enableVoice ? 'translate-x-6' : ''
                  }`}
                />
              </div>
            </div>

            {settings.enableVoice && (
              <>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Voice Rate
                  </label>
                  <input
                    type="range"
                    min="0.5"
                    max="2"
                    step="0.1"
                    value={settings.voiceRate}
                    onChange={(e) => handleSettingChange('voiceRate', parseFloat(e.target.value))}
                    className="w-full"
                  />
                  <div className="text-sm text-gray-500 mt-1">
                    {settings.voiceRate.toFixed(1)}x
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Voice Pitch
                  </label>
                  <input
                    type="range"
                    min="0.5"
                    max="2"
                    step="0.1"
                    value={settings.voicePitch}
                    onChange={(e) => handleSettingChange('voicePitch', parseFloat(e.target.value))}
                    className="w-full"
                  />
                  <div className="text-sm text-gray-500 mt-1">
                    {settings.voicePitch.toFixed(1)}
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default Settings; 