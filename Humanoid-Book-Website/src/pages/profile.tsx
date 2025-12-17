import React from 'react';
import Layout from '@theme/Layout';
import BrowserOnly from '@docusaurus/BrowserOnly';

function ProfilePageContent() {
  const { useState, useEffect } = React;
  const { useAuth } = require('../contexts/AuthContext');
  const { useAuthOperations } = require('../hooks/useAuthOperations');

  const { user, isAuthenticated, logout } = useAuth();
  const { loading: authLoading, fetchUserProfile, updateProfile } = useAuthOperations();
  const [profileData, setProfileData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState(false);
  const [formData, setFormData] = useState<any>({});
  const [updateLoading, setUpdateLoading] = useState(false);
  const [updateError, setUpdateError] = useState<string | null>(null);

  useEffect(() => {
    const loadProfile = async () => {
      if (isAuthenticated && user) {
        try {
          // Use the user data from context, or fetch fresh data from backend
          setProfileData(user);
          setFormData({
            first_name: user.first_name || '',
            last_name: user.last_name || '',
            email: user.email || '',
            // Add fields for user background info if available
            softwareExperience: user.softwareExperience || 'beginner',
            softwareDomain: user.softwareDomain || [],
            hardwareExperience: user.hardwareExperience || 'beginner',
            hardwareDomain: user.hardwareDomain || [],
            primaryProgrammingLanguage: user.primaryProgrammingLanguage || 'Python',
            yearsOfExperience: user.yearsOfExperience || 0,
            educationalBackground: user.educationalBackground || 'undergraduate',
            primaryGoal: user.primaryGoal || 'learn robotics'
          });
        } catch (error) {
          console.error('Error fetching profile:', error);
        }
      }
      setLoading(false);
    };

    loadProfile();
  }, [isAuthenticated, user]);

  if (authLoading || loading) {
    return (
      <Layout title="Loading" description="Loading your profile...">
        <div className="container margin-vert--lg">
          <div className="text--center">
            <h1>Loading Profile...</h1>
            <p>Please wait while we load your profile information.</p>
          </div>
        </div>
      </Layout>
    );
  }

  if (!isAuthenticated) {
    return (
      <Layout title="Profile" description="Please sign in to view your profile">
        <div className="container margin-vert--lg">
          <div className="row">
            <div className="col col--6 col--offset-3">
              <div className="card">
                <div className="card__header">
                  <h2>Access Denied</h2>
                </div>
                <div className="card__body">
                  <p>You need to be logged in to view your profile.</p>
                  <div className="button-group button-group--block">
                    <a className="button button--primary" href="/login">Sign In</a>
                    <a className="button button--secondary" href="/login">Create Account</a>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </Layout>
    );
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleCheckboxChange = (e: React.ChangeEvent<HTMLInputElement>, category: 'softwareDomain' | 'hardwareDomain') => {
    const value = e.target.value;
    setFormData(prev => {
      const currentValues = [...(prev[category] as string[]) || []];
      if (e.target.checked) {
        return {
          ...prev,
          [category]: [...currentValues, value]
        };
      } else {
        return {
          ...prev,
          [category]: currentValues.filter(item => item !== value)
        };
      }
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setUpdateLoading(true);
    setUpdateError(null);

    try {
      // Update profile on the backend
      const updatedProfile = await updateProfile({
        first_name: formData.first_name,
        last_name: formData.last_name,
        email: formData.email,
        // Include additional profile fields as needed
        software_experience: formData.softwareExperience,
        software_domains: formData.softwareDomain,
        hardware_experience: formData.hardwareExperience,
        hardware_domains: formData.hardwareDomain,
        primary_programming_language: formData.primaryProgrammingLanguage,
        years_experience: formData.yearsOfExperience,
        educational_background: formData.educationalBackground,
        primary_goal: formData.primaryGoal
      });

      // Update local state with new profile data
      setProfileData(updatedProfile.user);
      setFormData(prev => ({
        ...prev,
        first_name: updatedProfile.user.first_name,
        last_name: updatedProfile.user.last_name,
        email: updatedProfile.user.email,
        softwareExperience: updatedProfile.user.software_experience,
        softwareDomain: updatedProfile.user.software_domains,
        hardwareExperience: updatedProfile.user.hardware_experience,
        hardwareDomain: updatedProfile.user.hardware_domains,
        primaryProgrammingLanguage: updatedProfile.user.primary_programming_language,
        yearsOfExperience: updatedProfile.user.years_experience,
        educationalBackground: updatedProfile.user.educational_background,
        primaryGoal: updatedProfile.user.primary_goal
      }));
      setEditing(false);

      alert('Profile updated successfully!');
    } catch (error: any) {
      console.error('Error updating profile:', error);
      setUpdateError(error.message || 'Error updating profile. Please try again.');
    } finally {
      setUpdateLoading(false);
    }
  };

  const softwareDomains = [
    'web development',
    'mobile development',
    'AI/ML',
    'embedded systems',
    'robotics',
    'game development',
    'data science',
    'cybersecurity'
  ];

  const hardwareDomains = [
    'microcontrollers',
    'FPGAs',
    'sensors',
    'actuators',
    'PCB design',
    'embedded systems',
    'IoT',
    'robotics'
  ];

  return (
    <Layout title="User Profile" description="Manage your AI-Humanoid Robotics profile">
      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col col--8 col--offset-2">
            <div className="card">
              <div className="card__header">
                <h1>User Profile</h1>
              </div>

              <div className="card__body">
                <div style={{ marginBottom: '20px', textAlign: 'right' }}>
                  <button
                    onClick={logout}
                    className="button button--outline button--danger"
                  >
                    Sign Out
                  </button>
                </div>

                {updateError && (
                  <div className="alert alert--danger" role="alert">
                    {updateError}
                  </div>
                )}

                {!editing ? (
                  <div className="profile-content">
                    <h2>Personal Information</h2>
                    <p><strong>Name:</strong> {user?.first_name} {user?.last_name}</p>
                    <p><strong>Email:</strong> {user?.email}</p>

                    <h2>Software Background</h2>
                    <p><strong>Experience Level:</strong> {formData.softwareExperience || 'Not provided'}</p>
                    <p><strong>Domains:</strong> {formData.softwareDomain?.join(', ') || 'Not provided'}</p>

                    <h2>Hardware Background</h2>
                    <p><strong>Experience Level:</strong> {formData.hardwareExperience || 'Not provided'}</p>
                    <p><strong>Domains:</strong> {formData.hardwareDomain?.join(', ') || 'Not provided'}</p>

                    <h2>Additional Information</h2>
                    <p><strong>Primary Programming Language:</strong> {formData.primaryProgrammingLanguage || 'Not provided'}</p>
                    <p><strong>Years of Experience:</strong> {formData.yearsOfExperience || 'Not provided'}</p>
                    <p><strong>Educational Background:</strong> {formData.educationalBackground || 'Not provided'}</p>
                    <p><strong>Primary Goal:</strong> {formData.primaryGoal || 'Not provided'}</p>

                    <button
                      onClick={() => setEditing(true)}
                      className="button button--primary"
                    >
                      Edit Profile
                    </button>
                  </div>
                ) : (
                  <form onSubmit={handleSubmit} className="profile-edit-form">
                    <h2>Edit Profile</h2>

                    <div className="margin-bottom--md">
                      <label htmlFor="email" className="form-label">Email Address</label>
                      <input
                        type="email"
                        id="email"
                        name="email"
                        className="form-control"
                        value={formData.email || ''}
                        onChange={handleInputChange}
                        disabled // Email might be disabled depending on your auth system's rules
                      />
                    </div>

                    <div className="margin-bottom--md">
                      <label htmlFor="first_name" className="form-label">First Name</label>
                      <input
                        type="text"
                        id="first_name"
                        name="first_name"
                        className="form-control"
                        value={formData.first_name || ''}
                        onChange={handleInputChange}
                      />
                    </div>

                    <div className="margin-bottom--md">
                      <label htmlFor="last_name" className="form-label">Last Name</label>
                      <input
                        type="text"
                        id="last_name"
                        name="last_name"
                        className="form-control"
                        value={formData.last_name || ''}
                        onChange={handleInputChange}
                      />
                    </div>

                    <h3>Software Background</h3>
                    <div className="margin-bottom--md">
                      <label htmlFor="softwareExperience" className="form-label">Software Experience Level</label>
                      <select
                        id="softwareExperience"
                        name="softwareExperience"
                        className="form-control"
                        value={formData.softwareExperience}
                        onChange={handleInputChange}
                      >
                        <option value="beginner">Beginner</option>
                        <option value="intermediate">Intermediate</option>
                        <option value="advanced">Advanced</option>
                      </select>
                    </div>

                    <div className="margin-bottom--md">
                      <label className="form-label">Software Domains (select all that apply)</label>
                      {softwareDomains.map(domain => (
                        <div key={domain} className="form-checkbox">
                          <label className="checkbox">
                            <input
                              type="checkbox"
                              id={`software-${domain}`}
                              value={domain}
                              checked={(formData.softwareDomain || []).includes(domain)}
                              onChange={(e) => handleCheckboxChange(e, 'softwareDomain')}
                            />
                            <span>{domain}</span>
                          </label>
                        </div>
                      ))}
                    </div>

                    <h3>Hardware Background</h3>
                    <div className="margin-bottom--md">
                      <label htmlFor="hardwareExperience" className="form-label">Hardware Experience Level</label>
                      <select
                        id="hardwareExperience"
                        name="hardwareExperience"
                        className="form-control"
                        value={formData.hardwareExperience}
                        onChange={handleInputChange}
                      >
                        <option value="beginner">Beginner</option>
                        <option value="intermediate">Intermediate</option>
                        <option value="advanced">Advanced</option>
                      </select>
                    </div>

                    <div className="margin-bottom--md">
                      <label className="form-label">Hardware Domains (select all that apply)</label>
                      {hardwareDomains.map(domain => (
                        <div key={domain} className="form-checkbox">
                          <label className="checkbox">
                            <input
                              type="checkbox"
                              id={`hardware-${domain}`}
                              value={domain}
                              checked={(formData.hardwareDomain || []).includes(domain)}
                              onChange={(e) => handleCheckboxChange(e, 'hardwareDomain')}
                            />
                            <span>{domain}</span>
                          </label>
                        </div>
                      ))}
                    </div>

                    <div className="margin-bottom--md">
                      <label htmlFor="primaryProgrammingLanguage" className="form-label">Primary Programming Language</label>
                      <select
                        id="primaryProgrammingLanguage"
                        name="primaryProgrammingLanguage"
                        className="form-control"
                        value={formData.primaryProgrammingLanguage}
                        onChange={handleInputChange}
                      >
                        <option value="Python">Python</option>
                        <option value="C++">C++</option>
                        <option value="JavaScript">JavaScript</option>
                        <option value="TypeScript">TypeScript</option>
                        <option value="Java">Java</option>
                        <option value="C#">C#</option>
                        <option value="Rust">Rust</option>
                        <option value="Go">Go</option>
                        <option value="Other">Other</option>
                      </select>
                    </div>

                    <div className="margin-bottom--md">
                      <label htmlFor="yearsOfExperience" className="form-label">Years of Technical Experience</label>
                      <input
                        type="number"
                        id="yearsOfExperience"
                        name="yearsOfExperience"
                        className="form-control"
                        value={formData.yearsOfExperience}
                        onChange={handleInputChange}
                        min="0"
                        max="50"
                      />
                    </div>

                    <div className="margin-bottom--md">
                      <label htmlFor="educationalBackground" className="form-label">Educational Background</label>
                      <select
                        id="educationalBackground"
                        name="educationalBackground"
                        className="form-control"
                        value={formData.educationalBackground}
                        onChange={handleInputChange}
                      >
                        <option value="high school">High School</option>
                        <option value="undergraduate">Undergraduate</option>
                        <option value="graduate">Graduate</option>
                        <option value="postgraduate">Postgraduate</option>
                        <option value="self-taught">Self-taught</option>
                        <option value="bootcamp">Coding Bootcamp</option>
                        <option value="other">Other</option>
                      </select>
                    </div>

                    <div className="margin-bottom--md">
                      <label htmlFor="primaryGoal" className="form-label">Primary Goal</label>
                      <select
                        id="primaryGoal"
                        name="primaryGoal"
                        className="form-control"
                        value={formData.primaryGoal}
                        onChange={handleInputChange}
                      >
                        <option value="learn robotics">Learn Robotics</option>
                        <option value="career switch">Career Switch</option>
                        <option value="hobby">Hobby</option>
                        <option value="research">Research</option>
                        <option value="build product">Build Product</option>
                        <option value="other">Other</option>
                      </select>
                    </div>

                    <div className="button-group margin-top--md">
                      <button
                        type="submit"
                        className="button button--primary"
                        disabled={updateLoading}
                      >
                        {updateLoading ? 'Saving...' : 'Save Changes'}
                      </button>

                      <button
                        type="button"
                        className="button button--secondary"
                        onClick={() => setEditing(false)}
                        disabled={updateLoading}
                      >
                        Cancel
                      </button>
                    </div>
                  </form>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}

function ProfilePage() {
  return (
    <BrowserOnly>
      {() => <ProfilePageContent />}
    </BrowserOnly>
  );
}

export default ProfilePage;