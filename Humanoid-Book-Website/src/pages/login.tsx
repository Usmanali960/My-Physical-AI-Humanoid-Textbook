// src/pages/login.tsx
import React from 'react';
import Layout from '@theme/Layout';
import BrowserOnly from '@docusaurus/BrowserOnly';

function LoginPageContent() {
  const { useState } = React;
  const { useAuth } = require('../contexts/AuthContext');
  const { useAuthOperations } = require('../hooks/useAuthOperations');

  const { login } = useAuth();
  const { loading, error, login: loginUser, register, resetError } = useAuthOperations();
  const [loginForm, setLoginForm] = useState({ email: '', password: '' });
  const [showRegister, setShowRegister] = useState(false);
  const [registerForm, setRegisterForm] = useState({
    email: '',
    password: '',
    firstName: '',
    lastName: ''
  });

  const handleLoginChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setLoginForm(prev => ({ ...prev, [name]: value }));
    resetError();
  };

  const handleRegisterChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setRegisterForm(prev => ({ ...prev, [name]: value }));
    resetError();
  };

  const handleLoginSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const result = await loginUser(loginForm);
      login(result.user);
      // Redirect to profile or previous page
      window.location.href = '/profile';
    } catch (err) {
      console.error('Login failed:', err);
    }
  };

  const handleRegisterSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const userData = {
        email: registerForm.email,
        password: registerForm.password,
        first_name: registerForm.firstName,
        last_name: registerForm.lastName
      };

      // Register the user
      const result = await register(userData);

      // On successful registration, update the auth context with user data
      login(result.user || { email: registerForm.email, first_name: registerForm.firstName, last_name: registerForm.lastName });

      // Redirect to profile or previous page
      window.location.href = '/profile';
    } catch (err) {
      console.error('Registration failed:', err);
    }
  };

  return (
    <Layout title="Login" description="Login to your AI-Humanoid Robotics account">
      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col col--6 col--offset-3">
            <div className="card">
              <div className="card__header">
                <h2>{showRegister ? 'Create Account' : 'Login to Your Account'}</h2>
              </div>

              <div className="card__body">
                {error && (
                  <div className="alert alert--danger" role="alert">
                    {error}
                  </div>
                )}

                {showRegister ? (
                  <form onSubmit={handleRegisterSubmit}>
                    <div className="margin-bottom--md">
                      <label htmlFor="firstName" className="form-label">First Name</label>
                      <input
                        type="text"
                        id="firstName"
                        name="firstName"
                        className="form-control"
                        value={registerForm.firstName}
                        onChange={handleRegisterChange}
                        required
                      />
                    </div>

                    <div className="margin-bottom--md">
                      <label htmlFor="lastName" className="form-label">Last Name</label>
                      <input
                        type="text"
                        id="lastName"
                        name="lastName"
                        className="form-control"
                        value={registerForm.lastName}
                        onChange={handleRegisterChange}
                        required
                      />
                    </div>

                    <div className="margin-bottom--md">
                      <label htmlFor="regEmail" className="form-label">Email</label>
                      <input
                        type="email"
                        id="regEmail"
                        name="email"
                        className="form-control"
                        value={registerForm.email}
                        onChange={handleRegisterChange}
                        required
                      />
                    </div>

                    <div className="margin-bottom--md">
                      <label htmlFor="regPassword" className="form-label">Password</label>
                      <input
                        type="password"
                        id="regPassword"
                        name="password"
                        className="form-control"
                        value={registerForm.password}
                        onChange={handleRegisterChange}
                        required
                        minLength={8}
                      />
                    </div>

                    <button
                      type="submit"
                      className="button button--primary button--block"
                      disabled={loading}
                    >
                      {loading ? 'Creating Account...' : 'Sign Up'}
                    </button>
                  </form>
                ) : (
                  <form onSubmit={handleLoginSubmit}>
                    <div className="margin-bottom--md">
                      <label htmlFor="email" className="form-label">Email</label>
                      <input
                        type="email"
                        id="email"
                        name="email"
                        className="form-control"
                        value={loginForm.email}
                        onChange={handleLoginChange}
                        required
                      />
                    </div>

                    <div className="margin-bottom--md">
                      <label htmlFor="password" className="form-label">Password</label>
                      <input
                        type="password"
                        id="password"
                        name="password"
                        className="form-control"
                        value={loginForm.password}
                        onChange={handleLoginChange}
                        required
                      />
                    </div>

                    <button
                      type="submit"
                      className="button button--primary button--block"
                      disabled={loading}
                    >
                      {loading ? 'Logging In...' : 'Login'}
                    </button>
                  </form>
                )}

                <div className="margin-top--md">
                  <button
                    className="button button--outline button--block"
                    onClick={() => setShowRegister(!showRegister)}
                  >
                    {showRegister ? 'Already have an account? Login' : "Don't have an account? Sign Up"}
                  </button>
                </div>
              </div>
            </div>

            <div className="margin-top--lg text--center">
              <h3>Or sign in with</h3>
              <div className="button-group button-group--block margin-top--sm">
                {/* Google OAuth Button */}
                <button
                  className="button button--secondary"
                  onClick={() => {
                    // In a real implementation, this would trigger Google OAuth flow
                    alert('Google OAuth integration coming soon');
                  }}
                >
                  Sign in with Google
                </button>

                {/* GitHub OAuth Button */}
                <button
                  className="button button--secondary"
                  onClick={() => {
                    // In a real implementation, this would trigger GitHub OAuth flow
                    alert('GitHub OAuth integration coming soon');
                  }}
                >
                  Sign in with GitHub
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}

function LoginPage() {
  return (
    <BrowserOnly>
      {() => <LoginPageContent />}
    </BrowserOnly>
  );
}

export default LoginPage;