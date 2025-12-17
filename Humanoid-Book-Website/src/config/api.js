// src/config/api.js
// Configuration for the backend API

// Safely access environment variables in both Node.js and browser environments
const getBackendUrl = () => {
  // Check if we're in a browser environment where process is not defined
  if (typeof process === 'undefined' || !process.env) {
    // Fallback to a default URL when process is not available (browser environment)
    // This can be overridden by defining window.APP_CONFIG in index.html if needed
    return window.APP_CONFIG?.API_BASE_URL || 'http://127.0.0.1:8001';
  }
  // In Node.js environment, use process.env as usual
  return process.env.REACT_APP_API_URL ||
         process.env.REACT_APP_API_BASE_URL ||
         process.env.NEXT_PUBLIC_API_BASE_URL ||
         'http://127.0.0.1:8001';
};

// Base URL for the backend API
const BACKEND_API_URL = getBackendUrl();

// Export configuration object
export const API_CONFIG = {
  BASE_URL: BACKEND_API_URL,
  VERSION: 'v1',
  ENDPOINTS: {
    AUTH: {
      REGISTER: `${BACKEND_API_URL}/v1/auth/register`,
      LOGIN: `${BACKEND_API_URL}/v1/auth/login`,
      LOGOUT: `${BACKEND_API_URL}/v1/auth/logout`,
      PROFILE: `${BACKEND_API_URL}/v1/auth/profile`,
      OAUTH: `${BACKEND_API_URL}/v1/auth/oauth`,
    },
    TASKS: {
      LIST: `${BACKEND_API_URL}/v1/tasks`,
      DETAIL: (taskId) => `${BACKEND_API_URL}/v1/tasks/${taskId}`,
      EXECUTE: (taskId) => `${BACKEND_API_URL}/v1/tasks/${taskId}/execute`,
    },
    ADMIN: {
      USERS: `${BACKEND_API_URL}/v1/admin/users`,
      UPDATE_ROLE: (userId) => `${BACKEND_API_URL}/v1/admin/users/${userId}/role`,
      TASK_LOGS: `${BACKEND_API_URL}/v1/admin/tasks/logs`,
    },
  },
  HEADERS: {
    CONTENT_TYPE: 'application/json',
    AUTHORIZATION: (token) => `Bearer ${token}`,
  },
};

// Helper function to create authenticated request headers
export const getAuthHeaders = (token) => ({
  'Content-Type': API_CONFIG.HEADERS.CONTENT_TYPE,
  'Authorization': API_CONFIG.HEADERS.AUTHORIZATION(token),
});

export default API_CONFIG;