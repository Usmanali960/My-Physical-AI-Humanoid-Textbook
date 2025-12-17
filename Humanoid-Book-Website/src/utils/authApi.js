// src/utils/authApi.js
// Utility functions to interact with the backend authentication API
import { API_CONFIG } from '../config/api';

/**
 * Register a new user
 * @param {Object} userData - User registration data { email, password, first_name, last_name, software_experience, software_domains, hardware_experience, hardware_domains, primary_programming_language, years_experience, educational_background, primary_goal }
 * @returns {Promise<Object>} Registration response
 */
export const registerUser = async (userData) => {
  try {
    const response = await fetch(API_CONFIG.ENDPOINTS.AUTH.REGISTER, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(userData),
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || 'Registration failed');
    }

    // Store token in localStorage if provided in response (some backends return tokens on registration)
    if (result.access_token) {
      localStorage.setItem('auth_token', result.access_token);
    }

    return result;
  } catch (error) {
    console.error('Registration error:', error);
    throw error;
  }
};

/**
 * Login user
 * @param {Object} loginData - Login credentials { email, password }
 * @returns {Promise<Object>} Login response with token
 */
export const loginUser = async (loginData) => {
  try {
    const response = await fetch(API_CONFIG.ENDPOINTS.AUTH.LOGIN, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(loginData),
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || 'Login failed');
    }

    // Store token in localStorage
    if (result.access_token) {
      localStorage.setItem('auth_token', result.access_token);
    }

    return result;
  } catch (error) {
    console.error('Login error:', error);
    throw error;
  }
};

/**
 * OAuth login
 * @param {Object} oauthData - OAuth data { provider, oauth_token }
 * @returns {Promise<Object>} OAuth login response
 */
export const oauthLogin = async (oauthData) => {
  try {
    const response = await fetch(API_CONFIG.ENDPOINTS.AUTH.OAUTH, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(oauthData),
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || 'OAuth login failed');
    }

    // Store token in localStorage
    if (result.access_token) {
      localStorage.setItem('auth_token', result.access_token);
    }

    return result;
  } catch (error) {
    console.error('OAuth login error:', error);
    throw error;
  }
};

/**
 * Get current user profile
 * @returns {Promise<Object>} User profile data
 */
export const getUserProfile = async () => {
  try {
    const token = localStorage.getItem('auth_token');
    if (!token) {
      throw new Error('No authentication token found');
    }

    const response = await fetch(API_CONFIG.ENDPOINTS.AUTH.PROFILE, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
    });

    if (!response.ok) {
      if (response.status === 401) {
        localStorage.removeItem('auth_token');
        throw new Error('Authentication expired');
      }
      const result = await response.json();
      throw new Error(result.detail || 'Failed to fetch profile');
    }

    const result = await response.json();
    return result.user;
  } catch (error) {
    console.error('Get profile error:', error);
    // If there's a network error, clear the token to prevent repeated failed requests
    if (error.message.includes('Failed to fetch') ||
        error.message.includes('NetworkError') ||
        error.message.includes('token validation timeout') ||
        error.message.includes('profile fetch timeout')) {
      try {
        localStorage.removeItem('auth_token');
      } catch (storageError) {
        console.error('Error removing auth token:', storageError);
      }
      throw new Error('Unable to connect to authentication service');
    }
    throw error;
  }
};

/**
 * Update user profile
 * @param {Object} profileData - Profile update data
 * @returns {Promise<Object>} Update response
 */
export const updateUserProfile = async (profileData) => {
  try {
    const token = localStorage.getItem('auth_token');
    if (!token) {
      throw new Error('No authentication token found');
    }

    const response = await fetch(API_CONFIG.ENDPOINTS.AUTH.PROFILE, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify(profileData),
    });

    const result = await response.json();

    if (!response.ok) {
      if (response.status === 401) {
        localStorage.removeItem('auth_token');
        throw new Error('Authentication expired');
      }
      throw new Error(result.detail || 'Failed to update profile');
    }

    return result;
  } catch (error) {
    console.error('Update profile error:', error);
    throw error;
  }
};

/**
 * Logout user
 */
export const logoutUser = async () => {
  try {
    const token = localStorage.getItem('auth_token');

    if (token) {
      // Optionally notify backend about logout
      await fetch(API_CONFIG.ENDPOINTS.AUTH.LOGOUT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
      });
    }
  } catch (error) {
    console.error('Logout notification error:', error);
    // Still proceed with local cleanup even if backend notification fails
  } finally {
    // Remove token from localStorage
    localStorage.removeItem('auth_token');
  }
};

/**
 * Verify if token is still valid
 * @returns {Promise<boolean>} Whether token is valid
 */
export const verifyToken = async () => {
  try {
    const token = localStorage.getItem('auth_token');
    if (!token) {
      return false;
    }

    // Make a simple request to verify token
    const response = await fetch(API_CONFIG.ENDPOINTS.AUTH.PROFILE, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
      },
    });

    return response.ok;
  } catch (error) {
    console.error('Token verification error:', error);
    // Return false if there's a network error, but also clear the token
    // to prevent repeated failed requests
    try {
      localStorage.removeItem('auth_token');
    } catch (storageError) {
      console.error('Error removing auth token:', storageError);
    }
    return false;
  }
};

/**
 * Get auth headers with token
 * @returns {Object} Headers object with auth token
 */
export const getAuthHeaders = () => {
  const token = localStorage.getItem('auth_token');
  if (!token) {
    return {};
  }

  return {
    'Authorization': `Bearer ${token}`,
  };
};