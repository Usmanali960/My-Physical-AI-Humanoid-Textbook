// src/theme/wrapper.js
// This is a wrapper component that will wrap the entire app
import React from 'react';
import { AuthProvider } from '../contexts/AuthContext';

// Default theme wrapper component
export default function Wrapper({ children }) {
  return (
    <AuthProvider>
      {children}
    </AuthProvider>
  );
}