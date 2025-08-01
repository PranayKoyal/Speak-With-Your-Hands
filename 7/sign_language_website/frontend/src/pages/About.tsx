import React from 'react';
import Layout from '../components/layout/Layout';

export const About: React.FC = () => {
  return (
    <Layout>
      <div className="max-w-3xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">About Sign Language Recognition</h1>
        
        <div className="space-y-8">
          <section className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Project Overview</h2>
            <p className="text-gray-600 leading-relaxed">
              This sign language recognition system uses advanced computer vision and machine learning
              techniques to recognize and interpret sign language gestures in real-time. The project
              aims to bridge communication gaps and make sign language more accessible to everyone.
            </p>
          </section>

          <section className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Technology Stack</h2>
            <div className="grid grid-cols-2 gap-6">
              <div>
                <h3 className="font-medium mb-2">Frontend</h3>
                <ul className="list-disc list-inside text-gray-600 space-y-1">
                  <li>React with TypeScript</li>
                  <li>TailwindCSS</li>
                  <li>React Router</li>
                  <li>Web Speech API</li>
                </ul>
              </div>
              <div>
                <h3 className="font-medium mb-2">Backend</h3>
                <ul className="list-disc list-inside text-gray-600 space-y-1">
                  <li>Python Flask</li>
                  <li>OpenCV</li>
                  <li>TensorFlow/MediaPipe</li>
                  <li>WebSocket Communication</li>
                </ul>
              </div>
            </div>
          </section>

          <section className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">How It Works</h2>
            <div className="space-y-4 text-gray-600">
              <p>
                The system captures video input through your webcam and processes each frame to detect
                hand gestures. Using advanced machine learning models, it recognizes these gestures
                and converts them into text.
              </p>
              <p>
                The recognition process happens in real-time on the backend server, while the frontend
                provides an intuitive interface for interaction and displays the results. You can
                build sentences from recognized gestures and even have them spoken aloud using
                text-to-speech technology.
              </p>
            </div>
          </section>

          <section className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Privacy & Data</h2>
            <p className="text-gray-600 leading-relaxed">
              All video processing happens locally on your device and our server. We do not store
              or retain any video data. The system processes frames in real-time and discards them
              immediately after recognition.
            </p>
          </section>
        </div>
      </div>
    </Layout>
  );
};

export default About; 