// src/Root.tsx
import React from 'react';
import { AuthProvider } from './contexts/AuthContext';

// Root component that wraps the entire app with auth context
export default function Root({ children }: { children: React.ReactNode }) {
  return (
    <AuthProvider>
      {children}
    </AuthProvider>
  );
}