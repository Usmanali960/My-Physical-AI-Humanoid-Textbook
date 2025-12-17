/**
 * Content model for the AI-Humanoid Robotics book
 * 
 * This model represents the structure of book content that can be personalized
 * and potentially translated as per the specification requirements.
 */

export interface ContentElement {
  id: string;
  type: 'text' | 'code' | 'diagram' | 'example' | 'exercise' | 'summary' | 'definition';
  content: string;
  metadata?: {
    difficulty?: 'beginner' | 'intermediate' | 'advanced';
    targetAudience?: string[];  // e.g., ['software-dev', 'hardware-engineer', 'researcher']
    tags?: string[];            // e.g., ['ai', 'robotics', 'physical-ai']
  };
}

export interface Chapter {
  id: string;
  title: string;
  moduleId: string;
  position: number;
  description: string;
  content: ContentElement[];
  learningObjectives: string[];
  prerequisites?: string[];
  glossaryTerms?: string[];
  exercises?: Exercise[];
  metadata: {
    estimatedReadingTime: number;  // in minutes
    difficulty: 'beginner' | 'intermediate' | 'advanced';
    tags: string[];
    versions: ContentVersion[];    // For personalization and translation versions
  };
}

export interface ContentVersion {
  language: string;      // e.g., 'en', 'ur'
  variant: string;       // e.g., 'default', 'personalized-for-[user-type]'
  content: ContentElement[];
  generatedAt: Date;
  generatedBy: string;   // e.g., 'author', 'translation-ai', 'personalization-engine'
}

export interface Exercise {
  id: string;
  type: 'multiple-choice' | 'coding' | 'written' | 'practical';
  question: string;
  options?: string[];     // for multiple choice
  answer?: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  solution?: string;
  hints?: string[];
}

export interface BookModule {
  id: string;
  title: string;
  description: string;
  chapters: Chapter[];
  prerequisites?: string[];
  learningOutcomes: string[];
  estimatedCompletionTime: number; // in hours
}