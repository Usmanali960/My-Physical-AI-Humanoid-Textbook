/**
 * User model for storing user background information
 * 
 * This model captures the software and hardware background information 
 * as specified in the requirement FR-007: System MUST capture user's 
 * software and hardware background during signup
 */

export interface UserBackground {
  softwareExperience: string;        // e.g., "beginner", "intermediate", "advanced"
  softwareDomain: string[];          // e.g., ["web development", "mobile", "AI/ML", "embedded systems"]
  hardwareExperience: string;        // e.g., "beginner", "intermediate", "advanced"
  hardwareDomain: string[];          // e.g., ["microcontrollers", "FPGAs", "sensors", "actuators"]
  primaryProgrammingLanguage: string;// e.g., "Python", "C++", "JavaScript", etc.
  yearsOfExperience: number;         // Years of experience in tech
  educationalBackground: string;     // e.g., "high school", "undergraduate", "graduate", "self-taught"
  primaryGoal: string;               // e.g., "learn robotics", "career switch", "hobby", "research"
}

export interface User {
  id: string;
  email: string;
  name?: string;
  createdAt: Date;
  updatedAt: Date;
  background?: UserBackground;
  preferences?: {
    preferredLanguage?: string;      // e.g., "en", "ur"
    preferredContentLevel?: string;  // e.g., "beginner", "intermediate", "advanced"
    personalizationEnabled?: boolean;
  };
}