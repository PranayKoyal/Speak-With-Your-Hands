import React from 'react';
import { Link } from 'react-router-dom';
import Layout from '../components/layout/Layout';

export const Home: React.FC = () => {
  return (
    <Layout>
      <div className="max-w-4xl mx-auto text-center">
        <h1 className="text-4xl font-bold mb-6">
          Welcome to Sign Language Recognition
        </h1>
        <p className="text-xl text-gray-600 mb-8">
          Transform sign language into text and speech in real-time using your webcam.
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">How It Works</h2>
            <ul className="text-left space-y-3">
              <li className="flex items-start">
                <span className="mr-2">1.</span>
                <span>Allow access to your webcam</span>
              </li>
              <li className="flex items-start">
                <span className="mr-2">2.</span>
                <span>Perform sign language gestures in front of the camera</span>
              </li>
              <li className="flex items-start">
                <span className="mr-2">3.</span>
                <span>Watch as gestures are recognized and converted to text</span>
              </li>
              <li className="flex items-start">
                <span className="mr-2">4.</span>
                <span>Build sentences and hear them spoken aloud</span>
              </li>
            </ul>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Features</h2>
            <ul className="text-left space-y-3">
              <li className="flex items-start">
                <span className="mr-2">•</span>
                <span>Real-time sign language recognition</span>
              </li>
              <li className="flex items-start">
                <span className="mr-2">•</span>
                <span>Text-to-speech capabilities</span>
              </li>
              <li className="flex items-start">
                <span className="mr-2">•</span>
                <span>Adjustable recognition sensitivity</span>
              </li>
              <li className="flex items-start">
                <span className="mr-2">•</span>
                <span>Customizable voice settings</span>
              </li>
            </ul>
          </div>
        </div>
        
        <Link
          to="/recognition"
          className="inline-block bg-blue-600 text-white px-8 py-3 rounded-lg font-medium hover:bg-blue-700 transition-colors"
        >
          Get Started
        </Link>
      </div>
    </Layout>
  );
};

export default Home; 