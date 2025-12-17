import React from 'react';
import { AuthProvider } from '@site/src/contexts/AuthContext';

export default function Root({ children }) {
  return <AuthProvider>{children}</AuthProvider>;
}
