// src/config/api.js
// Configuration for the backend API

// Base URL for the backend API
const BACKEND_API_URL = process.env.REACT_APP_API_BASE_URL ||
                       process.env.NEXT_PUBLIC_API_BASE_URL ||
                       'http://localhost:8000';

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