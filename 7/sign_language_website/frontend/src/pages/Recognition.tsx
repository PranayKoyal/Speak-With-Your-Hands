import React from 'react';
import Layout from '../components/layout/Layout';
import VideoFeed from '../components/video/VideoFeed';
import VideoControls from '../components/video/VideoControls';
import GestureRecognition from '../components/recognition/GestureRecognition';
import SentenceBuilder from '../components/recognition/SentenceBuilder';

export const Recognition: React.FC = () => {
  return (
    <Layout>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="space-y-6">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h1 className="text-2xl font-bold mb-4">Sign Language Recognition</h1>
            <VideoFeed />
            <VideoControls />
          </div>
          <GestureRecognition />
        </div>
        <div>
          <SentenceBuilder />
        </div>
      </div>
    </Layout>
  );
};

export default Recognition; 