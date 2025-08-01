import React, { ReactNode } from 'react';
import Header from './Header';

interface LayoutProps {
  children: ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      <main className="container mx-auto px-4 py-8">
        {children}
      </main>
      <footer className="bg-white shadow-md mt-auto">
        <div className="container mx-auto px-4 py-4 text-center text-gray-600">
          <p>&copy; 2024 Sign Language Recognition. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
};

export default Layout; 