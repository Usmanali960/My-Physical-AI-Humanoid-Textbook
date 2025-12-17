// src/hooks/useAuthOperations.js
import { useState } from 'react';
import {
  registerUser,
  loginUser,
  oauthLogin,
  getUserProfile,
  updateUserProfile
} from '../utils/authApi';

// Custom hook for authentication operations
export const useAuthOperations = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Register a new user
  const register = async (userData) => {
    setLoading(true);
    setError(null);

    try {
      const result = await registerUser(userData);
      return result;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  // Login user
  const login = async (credentials) => {
    setLoading(true);
    setError(null);

    try {
      const result = await loginUser(credentials);
      return result;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  // OAuth login
  const oauthLoginHandler = async (oauthData) => {
    setLoading(true);
    setError(null);

    try {
      const result = await oauthLogin(oauthData);
      return result;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  // Get user profile
  const fetchUserProfile = async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await getUserProfile();
      return result;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  // Update user profile
  const updateProfile = async (profileData) => {
    setLoading(true);
    setError(null);

    try {
      const result = await updateUserProfile(profileData);
      return result;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  // Reset error
  const resetError = () => {
    setError(null);
  };

  return {
    loading,
    error,
    register,
    login,
    oauthLogin: oauthLoginHandler,
    fetchUserProfile,
    updateProfile,
    resetError
  };
};