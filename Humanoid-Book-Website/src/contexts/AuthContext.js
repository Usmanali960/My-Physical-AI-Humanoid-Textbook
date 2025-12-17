// src/contexts/AuthContext.js
import React, { createContext, useContext, useReducer, useEffect } from 'react';
import { verifyToken, logoutUser, getUserProfile } from '../utils/authApi';

// Check if we're in a browser environment
const isBrowser = typeof window !== 'undefined';

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
    // Only run authentication check in browser environment
    if (!isBrowser) {
      // On server, immediately complete initialization to avoid hanging
      dispatch({ type: 'CHECK_AUTH_STATUS_COMPLETE' });
      return;
    }

    let isMounted = true; // Track if component is still mounted

    const checkAuthStatus = async () => {
      try {
        if (!isMounted) return; // Prevent state updates if component unmounts

        dispatch({ type: 'CHECK_AUTH_STATUS_START' });

        // Check if token exists in localStorage before attempting validation
        const token = localStorage.getItem('auth_token');
        if (!token) {
          // No token exists, so user is not authenticated
          if (isMounted) dispatch({ type: 'CHECK_AUTH_STATUS_COMPLETE' });
          return;
        }

        // Attempt to validate the token with a timeout to prevent hanging
        const timeoutPromise = new Promise((_, reject) =>
          setTimeout(() => reject(new Error('Token validation timeout')), 5000)
        );

        let isValid = false;
        try {
          // Race the API call with a timeout to prevent indefinite hanging
          isValid = await Promise.race([
            verifyToken(),
            timeoutPromise
          ]);
        } catch (timeoutError) {
          console.error('Token verification timeout or error:', timeoutError.message);
        }

        if (isMounted && isValid) {
          try {
            // Token is valid, fetch user profile with timeout
            const profileTimeoutPromise = new Promise((_, reject) =>
              setTimeout(() => reject(new Error('Profile fetch timeout')), 5000)
            );

            const userProfile = await Promise.race([
              getUserProfile(),
              profileTimeoutPromise
            ]);

            if (isMounted) {
              dispatch({
                type: 'LOGIN_SUCCESS',
                payload: { user: userProfile }
              });
            }
          } catch (profileError) {
            console.error('Profile fetch error:', profileError.message);
            // Still continue to complete initialization even if profile fetch fails
          }
        } else if (isMounted && !isValid) {
          // Token is invalid, clear it from storage
          try {
            localStorage.removeItem('auth_token');
          } catch (storageError) {
            console.error('Error removing auth token:', storageError);
          }
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
      } finally {
        // Always dispatch completion to ensure the app continues to render
        if (isMounted) {
          dispatch({ type: 'CHECK_AUTH_STATUS_COMPLETE' });
        }
      }
    };

    // Execute the auth check, with error handling to prevent crashes
    checkAuthStatus().catch(error => {
      console.error('Unexpected error in auth check:', error);
      if (isMounted) {
        dispatch({ type: 'CHECK_AUTH_STATUS_COMPLETE' });
      }
    });

    // Cleanup function to prevent state updates on unmounted component
    return () => {
      isMounted = false;
    };
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

  // Ensure the app always renders children after initialization, regardless of auth state
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