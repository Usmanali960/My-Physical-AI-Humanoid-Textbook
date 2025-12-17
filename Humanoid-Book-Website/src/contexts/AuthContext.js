// src/contexts/AuthContext.js
import React, { createContext, useContext, useReducer, useEffect } from 'react';
import { verifyToken, logoutUser, getUserProfile } from '../utils/authApi';

// Create Auth Context
const AuthContext = createContext();

// Reducer for auth state management
const authReducer = (state, action) => {
  switch (action.type) {
    case 'LOGIN_START':
      return {
        ...state,
        loading: true,
        error: null,
      };
    case 'LOGIN_SUCCESS':
      return {
        ...state,
        loading: false,
        isAuthenticated: true,
        user: action.payload.user,
        error: null,
      };
    case 'LOGIN_FAILURE':
      return {
        ...state,
        loading: false,
        isAuthenticated: false,
        user: null,
        error: action.payload.error,
      };
    case 'LOGOUT':
      return {
        ...state,
        loading: false,
        isAuthenticated: false,
        user: null,
        error: null,
      };
    case 'SET_LOADING':
      return {
        ...state,
        loading: action.payload.loading,
      };
    case 'UPDATE_PROFILE':
      return {
        ...state,
        user: { ...state.user, ...action.payload.profile },
      };
    case 'CHECK_AUTH_STATUS_START':
      return {
        ...state,
        initializing: true,
      };
    case 'CHECK_AUTH_STATUS_COMPLETE':
      return {
        ...state,
        initializing: false,
      };
    default:
      return state;
  }
};

// Initial state
const initialState = {
  user: null,
  isAuthenticated: false,
  loading: false,
  error: null,
  initializing: true,
};

// Auth Provider Component
export const AuthProvider = ({ children }) => {
  const [state, dispatch] = useReducer(authReducer, initialState);

  // Check authentication status on app load
  useEffect(() => {
    const checkAuthStatus = async () => {
      try {
        dispatch({ type: 'CHECK_AUTH_STATUS_START' });

        // Check if token exists in localStorage before attempting validation
        const token = localStorage.getItem('auth_token');
        if (!token) {
          // No token exists, so user is not authenticated
          dispatch({ type: 'CHECK_AUTH_STATUS_COMPLETE' });
          return;
        }

        // Attempt to validate the token
        const isValid = await verifyToken();
        if (isValid) {
          // Token is valid, fetch user profile
          const userProfile = await getUserProfile();
          dispatch({
            type: 'LOGIN_SUCCESS',
            payload: { user: userProfile }
          });
        } else {
          // Token is invalid, clear it from storage
          localStorage.removeItem('auth_token');
        }
      } catch (error) {
        // Handle any errors during auth status check gracefully
        console.error('Auth verification error:', error);

        // Clear any potentially problematic token from storage
        try {
          localStorage.removeItem('auth_token');
        } catch (storageError) {
          console.error('Error removing auth token:', storageError);
        }

        // Ensure we complete the initialization regardless of errors
        dispatch({
          type: 'LOGIN_FAILURE',
          payload: { error: 'Authentication verification failed' }
        });
      } finally {
        // Always dispatch completion to ensure the app continues to render
        dispatch({ type: 'CHECK_AUTH_STATUS_COMPLETE' });
      }
    };

    // Execute the auth check, with error handling to prevent crashes
    checkAuthStatus().catch(error => {
      console.error('Unexpected error in auth check:', error);
      dispatch({ type: 'CHECK_AUTH_STATUS_COMPLETE' });
    });
  }, []);

  // Login handler
  const login = (userData) => {
    dispatch({
      type: 'LOGIN_SUCCESS',
      payload: { user: userData }
    });
  };

  // Logout handler
  const logout = async () => {
    try {
      await logoutUser();
      dispatch({ type: 'LOGOUT' });
    } catch (error) {
      console.error('Logout error:', error);
      // Still dispatch LOGOUT even if backend logout fails
      dispatch({ type: 'LOGOUT' });
    }
  };

  // Update user profile
  const updateProfile = (profileData) => {
    dispatch({
      type: 'UPDATE_PROFILE',
      payload: { profile: profileData }
    });
  };

  // Set loading state
  const setLoading = (isLoading) => {
    dispatch({
      type: 'SET_LOADING',
      payload: { loading: isLoading }
    });
  };

  return (
    <AuthContext.Provider value={{
      ...state,
      login,
      logout,
      updateProfile,
      setLoading
    }}>
      {state.initializing ? (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
          <p>Loading...</p>
        </div>
      ) : (
        children
      )}
    </AuthContext.Provider>
  );
};

// Hook to use auth context
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};