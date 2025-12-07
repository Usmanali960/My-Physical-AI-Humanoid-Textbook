---
id: module-06-chapter-02
title: Chapter 02 - Natural Language Processing for Robot Control
sidebar_position: 22
---

# Chapter 02 - Natural Language Processing for Robot Control

## Table of Contents
- [Overview](#overview)
- [Introduction to Natural Language in Robotics](#introduction-to-natural-language-in-robotics)
- [Language Understanding Challenges](#language-understanding-challenges)
- [Spatial Language Processing](#spatial-language-processing)
- [Intent Recognition and Classification](#intent-recognition-and-classification)
- [Semantic Parsing for Robot Commands](#semantic-parsing-for-robot-commands)
- [Context-Aware Language Processing](#context-aware-language-processing)
- [Multilingual Support](#multilingual-support)
- [Learning from Interaction](#learning-from-interaction)
- [Evaluation of Language Systems](#evaluation-of-language-systemes)
- [Future Directions](#future-directions)
- [Summary](#summary)
- [Exercises](#exercises)

## Overview

Natural Language Processing (NLP) for robot control enables robots to understand and respond to human commands expressed in everyday language. This capability is essential for making robots accessible to non-expert users and for creating natural, intuitive human-robot interactions. For humanoid robots, effective NLP is particularly important as these robots are designed to operate in human environments and interact with humans in familiar ways.

This chapter explores the specialized NLP techniques required for robot control, including understanding spatial references, handling ambiguous commands, and integrating linguistic information with sensor data. We'll examine the challenges of processing natural language in real-time, the importance of context awareness, and the role of learning in improving language understanding over time.

The chapter covers both traditional NLP approaches and modern deep learning methods, discussing their applications to robot control tasks. Special attention is given to spatial language processing, which is crucial for robots that must navigate and manipulate objects in 3D environments.

## Introduction to Natural Language in Robotics

### Natural Language Interface Requirements

Robots that accept natural language commands must handle a range of linguistic phenomena that traditional NLP applications might not encounter:

```python
# Key requirements for robot-directed language processing
ROBOT_LANGUAGE_REQUIREMENTS = {
    'spatial_language': {
        'requirement': 'Understanding spatial references and relationships',
        'examples': ['the cup on the table', 'behind the chair', 'to my left'],
        'challenge': 'Grounding spatial language to specific objects in the environment'
    },
    'deixis': {
        'requirement': 'Understanding pointing and demonstrative references',
        'examples': ['that one', 'this', 'the one I\'m pointing to'],
        'challenge': 'Determining referent based on pointing gestures and gaze'
    },
    'action_orientation': {
        'requirement': 'Understanding commands as action requests',
        'examples': ['pick up that book', 'go to the kitchen', 'open the door'],
        'challenge': 'Mapping linguistic actions to robot capabilities'
    },
    'context_dependence': {
        'requirement': 'Using context to disambiguate commands',
        'examples': ['do it again', 'like I showed you', 'the same way'],
        'challenge': 'Maintaining and using discourse context'
    },
    'real_time_processing': {
        'requirement': 'Processing language quickly enough for interactive use',
        'examples': 'Conversational turn-taking, interruptible commands',
        'challenge': 'Balancing accuracy with processing speed'
    }
}

class RobotLanguageProcessor:
    def __init__(self):
        self.speech_recognizer = SpeechRecognizer()
        self.intent_classifier = IntentClassifier()
        self.spatial_parser = SpatialLanguageParser()
        self.action_mapping = ActionMapper()
        self.context_manager = ContextManager()
        
    def process_command(self, audio_input, environment_context):
        """Process an audio command with environmental context"""
        # Step 1: Convert speech to text
        text_command = self.speech_recognizer.recognize(audio_input)
        
        # Step 2: Parse spatial language with environmental grounding
        spatial_info = self.spatial_parser.parse_with_grounding(
            text_command, environment_context
        )
        
        # Step 3: Classify intent
        intent = self.intent_classifier.classify(text_command)
        
        # Step 4: Map to robot action
        robot_action = self.action_mapping.map_to_robot(intent, spatial_info)
        
        # Step 5: Execute with context consideration
        execution_plan = self.context_manager.apply_context(
            robot_action, environment_context
        )
        
        return execution_plan
```

### Components of Robot Language Systems

A complete robot language system typically includes multiple interconnected components:

```cpp
// Robot language processing pipeline
class RobotLanguageSystem {
public:
    RobotLanguageSystem() {
        // Initialize all language processing components
        speech_recognizer_ = std::make_unique<SpeechRecognizer>();
        tokenizer_ = std::make_unique<Tokenizer>();
        parser_ = std::make_unique<SyntacticParser>();
        semantic_analyzer_ = std::make_unique<SemanticAnalyzer>();
        spatial_grounding_ = std::make_unique<SpatialGroundingModule>();
        intent_classifier_ = std::make_unique<IntentClassifier>();
        action_mapping_ = std::make_unique<ActionMapper>();
    }

    RobotCommand processLanguageCommand(const std::string& command,
                                       const EnvironmentContext& context) {
        // Tokenize command
        auto tokens = tokenizer_->tokenize(command);
        
        // Parse syntactic structure
        auto syntax_tree = parser_->parse(tokens);
        
        // Analyze semantics
        auto semantic_structure = semantic_analyzer_->analyze(syntax_tree);
        
        // Ground spatial references in the environment
        auto grounded_semantics = spatial_grounding_->ground(
            semantic_structure, context
        );
        
        // Classify user intent
        auto intent = intent_classifier_->classify(grounded_semantics);
        
        // Map to executable robot command
        auto robot_command = action_mapping_->mapToCommand(intent, context);
        
        return robot_command;
    }

private:
    std::unique_ptr<SpeechRecognizer> speech_recognizer_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<SyntacticParser> parser_;
    std::unique_ptr<SemanticAnalyzer> semantic_analyzer_;
    std::unique_ptr<SpatialGroundingModule> spatial_grounding_;
    std::unique_ptr<IntentClassifier> intent_classifier_;
    std::unique_ptr<ActionMapper> action_mapping_;
};
```

### Language-to-Action Mapping

Transforming natural language into robot actions requires careful consideration of the robot's capabilities:

```python
# Mapping language concepts to robot capabilities
LANGUAGE_ACTION_MAPPINGS = {
    'navigation': {
        'linguistic_patterns': [
            'go to', 'move to', 'walk to', 'navigate to', 'come here',
            'go to the', 'walk toward', 'move toward', 'go over there'
        ],
        'robot_actions': ['navigate_to', 'move_base', 'path_follow'],
        'spatial_requirements': ['destination_coordinates', 'path_planning']
    },
    'manipulation': {
        'linguistic_patterns': [
            'pick up', 'grasp', 'take', 'lift', 'hold', 'get',
            'put down', 'place', 'release', 'set down'
        ],
        'robot_actions': ['grasp_object', 'manipulate', 'place_object'],
        'spatial_requirements': ['object_pose', 'grasp_type', 'placement_location']
    },
    'social_interaction': {
        'linguistic_patterns': [
            'hello', 'goodbye', 'thank you', 'please', 'can you',
            'how are you', 'what can you do', 'tell me about'
        ],
        'robot_actions': ['greet', 'acknowledge', 'inform', 'answer_query'],
        'spatial_requirements': ['orientation_toward_speaker', 'facial_expression']
    }
}

class ActionMapper:
    def __init__(self):
        self.pattern_matcher = PatternMatcher()
        self.robot_capabilities = self.get_robot_capabilities()
        
    def map_to_robot(self, intent, spatial_info):
        """Map high-level intent to robot-executable action"""
        # Determine appropriate action based on intent type
        if intent.type == 'navigation':
            return self.create_navigation_action(intent, spatial_info)
        elif intent.type == 'manipulation':
            return self.create_manipulation_action(intent, spatial_info)
        elif intent.type == 'information_request':
            return self.create_information_action(intent)
        else:
            return self.create_default_action(intent)
            
    def create_navigation_action(self, intent, spatial_info):
        """Create navigation action from intent and spatial information"""
        if 'destination' in spatial_info:
            return {
                'action_type': 'navigate_to',
                'destination': spatial_info['destination'],
                'path_constraints': self.extract_path_constraints(intent),
                'speed_profile': self.determine_speed_profile(intent)
            }
        else:
            # If no specific destination, navigate to user
            return {
                'action_type': 'navigate_to_user',
                'user_reference': intent.user_reference,
                'preferred_distance': 1.0  # meter
            }
            
    def create_manipulation_action(self, intent, spatial_info):
        """Create manipulation action from intent and spatial information"""
        action_type = self.determine_manipulation_type(intent.verb)
        
        if 'target_object' in spatial_info:
            return {
                'action_type': action_type,
                'target_object': spatial_info['target_object'],
                'object_properties': spatial_info.get('object_properties', {}),
                'grasp_strategy': self.determine_grasp_strategy(
                    spatial_info['object_properties']
                )
            }
        else:
            raise ValueError("Manipulation command requires target object specification")
    
    def determine_manipulation_type(self, verb):
        """Determine manipulation action type from verb"""
        verb_action_map = {
            'pick up': 'grasp',
            'grasp': 'grasp', 
            'take': 'grasp',
            'lift': 'grasp',
            'hold': 'grasp',
            'get': 'grasp',
            'put down': 'place',
            'place': 'place',
            'release': 'release',
            'set down': 'place'
        }
        
        return verb_action_map.get(verb.lower(), 'grasp')
        
    def get_robot_capabilities(self):
        """Get robot's action capabilities"""
        # This would come from robot configuration
        return {
            'navigation': True,
            'manipulation': True,
            'speech_output': True,
            'facial_expressions': True,
            'social_gestures': True
        }
```

## Language Understanding Challenges

### Ambiguity Resolution

Natural language commands often contain ambiguities that must be resolved using context:

```python
class AmbiguityResolver:
    def __init__(self):
        self.coreference_resolver = CoreferenceResolver()
        self.spatial_resolver = SpatialReferenceResolver()
        self.world_knowledge = WorldKnowledgeBase()
        
    def resolve_ambiguities(self, command, context):
        """Resolve various types of ambiguities in the command"""
        resolved_command = command.copy()
        
        # Resolve pronouns and demonstratives
        resolved_command = self.resolve_coreferences(
            resolved_command, context
        )
        
        # Resolve spatial references
        resolved_command = self.resolve_spatial_references(
            resolved_command, context
        )
        
        # Resolve lexical ambiguities using world knowledge
        resolved_command = self.resolve_lexical_ambiguities(
            resolved_command, context
        )
        
        return resolved_command
        
    def resolve_coreferences(self, command, context):
        """Resolve pronouns and demonstratives"""
        # Identify pronouns and demonstratives in the command
        pronouns = self.find_pronouns(command)
        demonstratives = self.find_demonstratives(command)
        
        # Resolve using context
        for pronoun in pronouns:
            referent = self.determine_pronoun_referent(pronoun, context)
            command = self.replace_pronoun(command, pronoun, referent)
            
        for demo in demonstratives:
            referent = self.determine_demonstrative_referent(demo, context)
            command = self.replace_demonstrative(command, demo, referent)
            
        return command
        
    def find_pronouns(self, command):
        """Find pronouns in the command"""
        pronouns = ['it', 'they', 'them', 'this', 'that', 'these', 'those']
        found = []
        
        for word in command.split():
            if word.lower() in pronouns:
                found.append(word)
                
        return found
        
    def find_demonstratives(self, command):
        """Find demonstratives in the command"""
        demonstratives = ['this', 'that', 'these', 'those']
        found = []
        
        for word in command.split():
            if word.lower() in demonstratives:
                found.append(word)
                
        return found
        
    def determine_pronoun_referent(self, pronoun, context):
        """Determine what a pronoun refers to in context"""
        if pronoun.lower() == 'it':
            # Often refers to the most recently mentioned object
            if context.recent_objects:
                return context.recent_objects[-1]
        elif pronoun.lower() in ['they', 'them']:
            # Often refers to multiple recently mentioned objects
            if len(context.recent_objects) >= 2:
                return context.recent_objects[-2:]
                
        # Default to asking for clarification
        return 'UNCERTAIN_REFERENT'
        
    def resolve_spatial_references(self, command, context):
        """Resolve spatial references like 'left', 'right', 'there'"""
        # Identify spatial expressions
        spatial_expressions = self.find_spatial_expressions(command)
        
        for expr in spatial_expressions:
            # Resolve relative to robot or user perspective
            resolved_location = self.resolve_spatial_expression(
                expr, context
            )
            command = self.replace_spatial_reference(
                command, expr, resolved_location
            )
            
        return command
        
    def find_spatial_expressions(self, command):
        """Find spatial expressions in command"""
        spatial_terms = [
            'left', 'right', 'front', 'back', 'behind', 'in front of',
            'next to', 'near', 'there', 'here', 'above', 'below'
        ]
        
        found = []
        for term in spatial_terms:
            if term in command.lower():
                found.append(term)
                
        return found
```

### Handling Language Variations

Robots must understand diverse ways of expressing the same command:

```cpp
// Language variation handler for robot commands
class LanguageVariationHandler {
public:
    LanguageVariationHandler() {
        initializeVariationMappings();
    }

    std::string normalizeCommand(const std::string& command) {
        // Normalize different ways of expressing the same concept
        std::string normalized = command;
        
        // Synonym mapping
        normalized = applySynonymMapping(normalized);
        
        // Paraphrase normalization
        normalized = applyParaphraseNormalization(normalized);
        
        // Colloquialism handling
        normalized = applyColloquialismHandling(normalized);
        
        return normalized;
    }

    std::vector<CommandVariant> findVariants(const std::string& base_command) {
        std::vector<CommandVariant> variants;
        
        // Generate synonyms
        auto synonym_variants = generateSynonymVariants(base_command);
        variants.insert(variants.end(), 
                       synonym_variants.begin(), synonym_variants.end());
        
        // Generate paraphrase variants
        auto paraphrase_variants = generateParaphraseVariants(base_command);
        variants.insert(variants.end(),
                       paraphrase_variants.begin(), paraphrase_variants.end());
        
        // Generate colloquial variants
        auto colloquial_variants = generateColloquialVariants(base_command);
        variants.insert(variants.end(),
                       colloquial_variants.begin(), colloquial_variants.end());
        
        return variants;
    }

private:
    struct CommandVariant {
        std::string text;
        double similarity_score;
        std::string canonical_form;
        std::vector<std::string> equivalent_intents;
    };

    void initializeVariationMappings() {
        // Map different expressions to canonical forms
        synonym_map_ = {
            {"pick up", "grasp"},
            {"take", "grasp"},
            {"get", "grasp"},
            {"lift", "grasp"},
            {"bring", "transport"},
            {"go to", "navigate_to"},
            {"move to", "navigate_to"},
            {"walk to", "navigate_to"},
            {"put", "place"},
            {"set", "place"},
            {"place", "place"},
            {"drop", "release"}
        };
        
        paraphrase_map_ = {
            {"Could you please bring me the red cup?", "Bring the red cup"},
            {"Would you mind fetching the book from the table?", "Fetch the book from the table"},
            {"I need you to go to the kitchen", "Go to the kitchen"},
            {"Can you please open the door for me?", "Open the door"}
        };
        
        colloquial_map_ = {
            {"fetch", "get"},
            {"grab", "take"},
            {"head to", "go to"},
            {"make your way to", "go to"},
            {"slide over to", "go to"},
            {"hustle over to", "go to"}
        };
    }

    std::string applySynonymMapping(const std::string& command) {
        std::string result = command;
        
        for (const auto& [variation, canonical] : synonym_map_) {
            size_t pos = result.find(variation);
            if (pos != std::string::npos) {
                result.replace(pos, variation.length(), canonical);
            }
        }
        
        return result;
    }
    
    std::string applyParaphraseNormalization(const std::string& command) {
        // Look for known paraphrases and normalize to canonical form
        for (const auto& [paraphrase, canonical] : paraphrase_map_) {
            if (command == paraphrase) {
                return canonical;
            }
        }
        
        return command;
    }
    
    std::string applyColloquialismHandling(const std::string& command) {
        std::string result = command;
        
        for (const auto& [colloquial, standard] : colloquial_map_) {
            size_t pos = result.find(colloquial);
            if (pos != std::string::npos) {
                result.replace(pos, colloquial.length(), standard);
            }
        }
        
        return result;
    }
    
    std::vector<CommandVariant> generateSynonymVariants(const std::string& base_command) {
        std::vector<CommandVariant> variants;
        
        // Generate variants by substituting synonyms
        for (const auto& [variation, canonical] : synonym_map_) {
            if (base_command.find(canonical) != std::string::npos) {
                std::string variant = base_command;
                size_t pos = variant.find(canonical);
                variant.replace(pos, canonical.length(), variation);
                
                variants.push_back({
                    .text = variant,
                    .similarity_score = 0.8,
                    .canonical_form = base_command,
                    .equivalent_intents = {getIntentForCommand(base_command)}
                });
            }
        }
        
        return variants;
    }
    
    std::vector<CommandVariant> generateParaphraseVariants(const std::string& base_command) {
        std::vector<CommandVariant> variants;
        
        // For this example, just return the base command as canonical
        variants.push_back({
            .text = base_command,
            .similarity_score = 1.0,
            .canonical_form = base_command,
            .equivalent_intents = {getIntentForCommand(base_command)}
        });
        
        return variants;
    }
    
    std::vector<CommandVariant> generateColloquialVariants(const std::string& base_command) {
        std::vector<CommandVariant> variants;
        
        // Generate variants using colloquial expressions
        for (const auto& [colloquial, standard] : colloquial_map_) {
            if (base_command.find(standard) != std::string::npos) {
                std::string variant = base_command;
                size_t pos = variant.find(standard);
                variant.replace(pos, standard.length(), colloquial);
                
                variants.push_back({
                    .text = variant,
                    .similarity_score = 0.7,
                    .canonical_form = base_command,
                    .equivalent_intents = {getIntentForCommand(base_command)}
                });
            }
        }
        
        return variants;
    }
    
    std::string getIntentForCommand(const std::string& command) {
        // Determine intent from command text
        if (command.find("grasp") != std::string::npos ||
            command.find("take") != std::string::npos ||
            command.find("get") != std::string::npos) {
            return "manipulation";
        } else if (command.find("navigate") != std::string::npos ||
                   command.find("go to") != std::string::npos) {
            return "navigation";
        } else {
            return "other";
        }
    }
    
    std::map<std::string, std::string> synonym_map_;
    std::map<std::string, std::string> paraphrase_map_;
    std::map<std::string, std::string> colloquial_map_;
};
```

## Spatial Language Processing

### Understanding Spatial References

Spatial language is fundamental to robot control, as robots must operate in 3D environments:

```python
# Spatial language processing for robotics
SPATIAL_RELATIONS = {
    'projective': ['left', 'right', 'front', 'back'],  # Relative to perspective
    'absolute': ['north', 'south', 'east', 'west'],   # Fixed directions
    'relative': ['above', 'below', 'beside', 'behind', 'in front of'],
    'topological': ['in', 'on', 'under', 'next to', 'between']
}

class SpatialLanguageProcessor:
    def __init__(self):
        self.spatial_parser = SpatialParser()
        self.reference_resolver = SpatialReferenceResolver()
        self.grounding_system = EnvironmentalGroundingSystem()
        
    def process_spatial_command(self, command, environment):
        """Process a command with spatial components"""
        # Parse spatial language components
        spatial_elements = self.spatial_parser.extract_elements(command)
        
        # Resolve spatial references to environment objects
        resolved_elements = self.resolve_spatial_references(
            spatial_elements, environment
        )
        
        # Ground spatial relations in the environment
        grounded_relations = self.grounding_system.ground_relations(
            resolved_elements, environment
        )
        
        return {
            'action_components': self.extract_action_components(command),
            'spatial_constraints': grounded_relations,
            'target_location': self.determine_target_location(
                grounded_relations, environment
            )
        }
        
    def resolve_spatial_references(self, elements, environment):
        """Resolve spatial references to environmental entities"""
        resolved = {}
        
        for element_type, element_value in elements.items():
            if element_type == 'object_reference':
                resolved[element_type] = self.resolve_object_reference(
                    element_value, environment
                )
            elif element_type == 'location_reference':
                resolved[element_type] = self.resolve_location_reference(
                    element_value, environment
                )
            elif element_type == 'spatial_relation':
                resolved[element_type] = self.resolve_spatial_relation(
                    element_value, environment
                )
                
        return resolved
        
    def resolve_object_reference(self, reference, environment):
        """Resolve object reference to specific environmental object"""
        # Handle definite descriptions
        if 'the' in reference:
            # Find the specific object matching the description
            description = reference.replace('the ', '').strip()
            return self.find_best_matching_object(description, environment)
        elif reference.lower() in ['it', 'this', 'that']:
            # Resolve based on context or pointing
            return self.resolve_demonstrative(reference, environment)
        else:
            # Look for object by name/type
            return self.find_object_by_type(reference, environment)
            
    def find_best_matching_object(self, description, environment):
        """Find the most likely object matching the description"""
        candidates = []
        
        for obj in environment.objects:
            # Calculate match score based on properties
            score = self.calculate_match_score(description, obj)
            candidates.append((obj, score))
            
        # Return best match
        if candidates:
            best = max(candidates, key=lambda x: x[1])
            return best[0] if best[1] > 0.3 else None  # threshold
        else:
            return None
            
    def calculate_match_score(self, description, obj):
        """Calculate how well an object matches a description"""
        score = 0.0
        
        # Match on object type
        if description.lower() in obj.type.lower():
            score += 0.4
            
        # Match on color
        if hasattr(obj, 'color') and obj.color.lower() in description.lower():
            score += 0.3
            
        # Match on spatial attributes
        if 'big' in description.lower() and obj.size == 'large':
            score += 0.2
        elif 'small' in description.lower() and obj.size == 'small':
            score += 0.2
            
        # Match on location keywords
        if 'table' in description.lower() and obj.location == 'on_table':
            score += 0.1
            
        return score
        
    def resolve_spatial_relation(self, relation, environment):
        """Resolve spatial relationship between entities"""
        # Parse relation components
        rel_parts = relation.split()
        
        if len(rel_parts) >= 3:
            obj1, spatial_preposition, obj2 = rel_parts[0], rel_parts[1], rel_parts[2]
            
            # Resolve the two objects
            resolved_obj1 = self.resolve_object_reference(obj1, environment)
            resolved_obj2 = self.resolve_object_reference(obj2, environment)
            
            # Calculate spatial relationship
            if resolved_obj1 and resolved_obj2:
                relationship = self.calculate_spatial_relationship(
                    resolved_obj1, resolved_obj2, spatial_preposition
                )
                return relationship
                
        return None

class SpatialParser:
    def extract_elements(self, command):
        """Extract spatial elements from a command"""
        elements = {
            'object_references': self.find_object_references(command),
            'location_references': self.find_location_references(command),
            'spatial_relations': self.find_spatial_relations(command),
            'spatial_deictics': self.find_spatial_deictics(command)
        }
        
        return elements
        
    def find_object_references(self, command):
        """Find object references in the command"""
        # Look for noun phrases that likely refer to objects
        import re
        
        # Pattern for object references: [determiner] [adjective]* [noun]
        pattern = r'\b(the|a|an)\s+((?:\w+\s+)*?)(?:cup|book|chair|table|door|box|mug|bottle|object|item|thing)\b'
        matches = re.finditer(pattern, command, re.IGNORECASE)
        
        references = []
        for match in matches:
            full_match = match.group(0)
            references.append(full_match.strip())
            
        return references
        
    def find_spatial_relations(self, command):
        """Find spatial relationship expressions"""
        relations = []
        
        # Look for prepositional phrases indicating spatial relations
        for rel in SPATIAL_RELATIONS['relative'] + SPATIAL_RELATIONS['topological']:
            if rel in command.lower():
                relations.append(rel)
                
        return relations
        
    def find_spatial_deictics(self, command):
        """Find spatial deictic expressions ('here', 'there', 'this', 'that')"""
        deictics = ['here', 'there', 'this', 'that']
        found = []
        
        for deictic in deictics:
            if deictic in command.lower():
                found.append(deictic)
                
        return found
```

### Perspective and Reference Frames

Robots must handle different reference frames (robot-centered, user-centered, environment-centered):

```cpp
class ReferenceFrameHandler {
public:
    ReferenceFrameHandler() {
        robot_pose_ = {0, 0, 0, 0}; // x, y, z, theta
        user_pose_ = {0, 0, 0, 0};
        updateTransforms();
    }

    SpatialReference resolveWithReferenceFrame(const SpatialReference& reference,
                                             const std::string& reference_frame) {
        if (reference_frame == "robot") {
            return resolveRelativeToRobot(reference);
        } else if (reference_frame == "user") {
            return resolveRelativeToUser(reference);
        } else if (reference_frame == "environment") {
            return reference; // Already in environment frame
        } else {
            // Default to robot frame
            return resolveRelativeToRobot(reference);
        }
    }

    SpatialLocation transformToEnvironmentFrame(const SpatialReference& reference,
                                              const std::string& source_frame) {
        if (source_frame == "robot") {
            return transformRobotToEnvironment(reference);
        } else if (source_frame == "user") {
            return transformUserToEnvironment(reference);
        } else {
            return reference.location; // Already in environment frame
        }
    }

private:
    Pose robot_pose_;
    Pose user_pose_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    void updateTransforms() {
        try {
            // Get robot pose in environment frame
            geometry_msgs::TransformStamped robot_transform = 
                tf_buffer_.lookupTransform("map", "base_link", ros::Time(0));
            robot_pose_ = transformToPose(robot_transform);

            // Get user pose in environment frame (assuming user is tracked)
            geometry_msgs::TransformStamped user_transform = 
                tf_buffer_.lookupTransform("map", "user_frame", ros::Time(0));
            user_pose_ = transformToPose(user_transform);
        } catch (tf2::TransformException &ex) {
            ROS_WARN("Could not get transform: %s", ex.what());
        }
    }

    SpatialReference resolveRelativeToRobot(const SpatialReference& reference) {
        // Transform relative directions (left, right, front, back) 
        // to absolute coordinates based on robot orientation
        SpatialReference resolved = reference;
        
        if (reference.direction == "left") {
            resolved.location = calculateLeftOfRobot(reference.distance);
        } else if (reference.direction == "right") {
            resolved.location = calculateRightOfRobot(reference.distance);
        } else if (reference.direction == "front" || reference.direction == "forward") {
            resolved.location = calculateFrontOfRobot(reference.distance);
        } else if (reference.direction == "back") {
            resolved.location = calculateBackOfRobot(reference.distance);
        }
        
        return resolved;
    }

    SpatialReference resolveRelativeToUser(const SpatialReference& reference) {
        // Transform relative to user's perspective
        SpatialReference resolved = reference;
        
        if (reference.direction == "left") {
            resolved.location = calculateLeftOfUser(reference.distance);
        } else if (reference.direction == "right") {
            resolved.location = calculateRightOfUser(reference.distance);
        } else if (reference.direction == "front" || reference.direction == "forward") {
            resolved.location = calculateFrontOfUser(reference.distance);
        } else if (reference.direction == "back") {
            resolved.location = calculateBackOfUser(reference.distance);
        }
        
        return resolved;
    }

    Point3D calculateLeftOfRobot(double distance) {
        // Calculate point to the left of robot based on robot orientation
        double robot_angle = robot_pose_.theta;
        double x = robot_pose_.x + distance * cos(robot_angle + M_PI/2);
        double y = robot_pose_.y + distance * sin(robot_angle + M_PI/2);
        return {x, y, robot_pose_.z};
    }

    Point3D calculateRightOfRobot(double distance) {
        // Calculate point to the right of robot based on robot orientation
        double robot_angle = robot_pose_.theta;
        double x = robot_pose_.x + distance * cos(robot_angle - M_PI/2);
        double y = robot_pose_.y + distance * sin(robot_angle - M_PI/2);
        return {x, y, robot_pose_.z};
    }

    Point3D calculateFrontOfRobot(double distance) {
        // Calculate point in front of robot based on robot orientation
        double robot_angle = robot_pose_.theta;
        double x = robot_pose_.x + distance * cos(robot_angle);
        double y = robot_pose_.y + distance * sin(robot_angle);
        return {x, y, robot_pose_.z};
    }

    Point3D calculateBackOfRobot(double distance) {
        // Calculate point behind robot based on robot orientation
        double robot_angle = robot_pose_.theta;
        double x = robot_pose_.x + distance * cos(robot_angle + M_PI);
        double y = robot_pose_.y + distance * sin(robot_angle + M_PI);
        return {x, y, robot_pose_.z};
    }

    Pose transformToPose(const geometry_msgs::TransformStamped& transform) {
        // Convert transform to pose structure
        Pose pose;
        pose.x = transform.transform.translation.x;
        pose.y = transform.transform.translation.y;
        pose.z = transform.transform.translation.z;
        
        // Convert quaternion to euler angle for 2D orientation
        tf2::Quaternion q(
            transform.transform.rotation.x,
            transform.transform.rotation.y, 
            transform.transform.rotation.z,
            transform.transform.rotation.w
        );
        tf2::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);
        pose.theta = yaw;
        
        return pose;
    }
    
    Point3D calculateLeftOfUser(double distance);
    Point3D calculateRightOfUser(double distance);
    Point3D calculateFrontOfUser(double distance);
    Point3D calculateBackOfUser(double distance);
    
    Point3D transformRobotToEnvironment(const SpatialReference& reference);
    Point3D transformUserToEnvironment(const SpatialReference& reference);
};
```

## Intent Recognition and Classification

### Hierarchical Intent Classification

Intent recognition for robotics often requires hierarchical classification to handle complex command structures:

```python
# Hierarchical intent classification for robot commands
INTENT_HIERARCHY = {
    'navigation': {
        'goto_location': ['go to', 'move to', 'navigate to', 'walk to'],
        'explore': ['explore', 'look around', 'scan area'],
        'follow': ['follow', 'come with me', 'accompany']
    },
    'manipulation': {
        'grasp_object': ['pick up', 'take', 'grasp', 'get', 'lift'],
        'place_object': ['place', 'put', 'set', 'put down'],
        'transport_object': ['bring', 'carry', 'move', 'deliver'],
        'interact_with_object': ['open', 'close', 'press', 'push', 'pull']
    },
    'social_interaction': {
        'greeting': ['hello', 'hi', 'good morning', 'good evening'],
        'farewell': ['goodbye', 'bye', 'see you', 'farewell'],
        'answer_query': ['what', 'how', 'where', 'when', 'who'],
        'provide_information': ['tell', 'explain', 'describe', 'inform']
    },
    'system_control': {
        'start': ['start', 'begin', 'initiate'],
        'stop': ['stop', 'halt', 'cease', 'pause'],
        'reset': ['reset', 'restart', 'reinitialize']
    }
}

class HierarchicalIntentClassifier:
    def __init__(self):
        self.keyword_matcher = KeywordMatcher(INTENT_HIERARCHY)
        self.pattern_matcher = PatternMatcher()
        self.context_aware_classifier = ContextAwareClassifier()
        
    def classify_intent(self, command, context=None):
        """Classify intent hierarchically"""
        # Step 1: Use keyword matching for initial classification
        primary_candidates = self.keyword_matcher.match_keywords(command)
        
        # Step 2: Use pattern matching for more nuanced classification
        pattern_candidates = self.pattern_matcher.match_patterns(command)
        
        # Step 3: Apply context awareness
        final_intent = self.context_aware_classifier.disambiguate(
            primary_candidates, pattern_candidates, command, context
        )
        
        return final_intent
        
    def classify_with_confidence(self, command, context=None):
        """Classify intent with confidence scores"""
        # Get all possible intent matches with scores
        matches = self.get_all_intent_matches(command)
        
        # Apply context weighting
        if context:
            matches = self.apply_context_weights(matches, context)
            
        # Sort by final score
        sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_matches:
            best_intent, confidence = sorted_matches[0]
            return {
                'primary': best_intent,
                'confidence': confidence,
                'alternatives': [
                    {'intent': intent, 'confidence': score} 
                    for intent, score in sorted_matches[1:4]  # Top 3 alternatives
                ]
            }
        else:
            return {
                'primary': 'unknown',
                'confidence': 0.0,
                'alternatives': []
            }
    
    def get_all_intent_matches(self, command):
        """Get all possible intent matches with scores"""
        matches = {}
        
        # Match against all patterns in hierarchy
        for super_category, sub_categories in INTENT_HIERARCHY.items():
            for sub_category, patterns in sub_categories.items():
                score = self.calculate_match_score(command, patterns)
                if score > 0.1:  # Threshold to include match
                    matches[f"{super_category}.{sub_category}"] = score
                    
        return matches
        
    def calculate_match_score(self, command, patterns):
        """Calculate match score for command against patterns"""
        score = 0.0
        
        for pattern in patterns:
            # Simple substring matching with length normalization
            if pattern.lower() in command.lower():
                pattern_score = len(pattern) / len(command)
                score = max(score, pattern_score)
                
        # Enhance score for exact phrase matches
        for pattern in patterns:
            if f" {pattern.lower()} " in f" {command.lower()} ":
                score += 0.2  # Bonus for phrase matches
                
        return min(score, 1.0)  # Cap at 1.0
        
    def apply_context_weights(self, matches, context):
        """Apply weights based on context information"""
        weighted_matches = matches.copy()
        
        # Increase weight if command matches recent context
        if hasattr(context, 'recent_interactions'):
            for intent, score in weighted_matches.items():
                for recent in context.recent_interactions[-3:]:  # Last 3 interactions
                    if recent.intent == intent:
                        weighted_matches[intent] += 0.1  # Context bonus
                        
        # Increase weight if object in command matches context
        if hasattr(context, 'available_objects'):
            object_names = [obj.name.lower() for obj in context.available_objects]
            for intent, score in weighted_matches.items():
                for obj_name in object_names:
                    if obj_name in intent:
                        weighted_matches[intent] += 0.15  # Object context bonus
                        
        return weighted_matches

class KeywordMatcher:
    def __init__(self, intent_hierarchy):
        self.intent_hierarchy = intent_hierarchy
        self.keyword_mappings = self.build_keyword_mappings()
        
    def build_keyword_mappings(self):
        """Build mapping from keywords to intents"""
        mappings = {}
        
        for super_category, sub_categories in self.intent_hierarchy.items():
            for sub_category, keywords in sub_categories.items():
                for keyword in keywords:
                    intent_path = f"{super_category}.{sub_category}"
                    if keyword.lower() not in mappings:
                        mappings[keyword.lower()] = []
                    mappings[keyword.lower()].append(intent_path)
                    
        return mappings
        
    def match_keywords(self, command):
        """Match keywords in command to intents"""
        matches = []
        command_lower = command.lower()
        
        for keyword, intents in self.keyword_mappings.items():
            if keyword in command_lower:
                for intent in intents:
                    matches.append({
                        'intent': intent,
                        'keyword': keyword,
                        'position': command_lower.find(keyword),
                        'confidence': self.calculate_keyword_confidence(keyword, command)
                    })
                    
        return matches
        
    def calculate_keyword_confidence(self, keyword, command):
        """Calculate confidence based on keyword properties"""
        # Simple scoring based on keyword position and length
        pos = command.lower().find(keyword)
        if pos == 0:  # At beginning
            return 0.8
        elif pos < 10:  # Early in command
            return 0.7
        else:
            return 0.6  # Later in command

class ContextAwareClassifier:
    def __init__(self):
        self.context_model = ContextModel()
        
    def disambiguate(self, primary_candidates, pattern_candidates, command, context):
        """Disambiguate between multiple intent candidates using context"""
        if len(set(c['intent'] for c in primary_candidates)) == 1:
            # If all candidates point to same intent, return it
            return primary_candidates[0]['intent']
        elif len(primary_candidates) == 0 and len(pattern_candidates) == 0:
            # If no clear matches, try to determine from general features
            return self.guess_intent_from_features(command, context)
        else:
            # Use context to select best candidate
            return self.select_best_candidate(
                primary_candidates, pattern_candidates, context
            )
            
    def select_best_candidate(self, primary_candidates, pattern_candidates, context):
        """Select best candidate based on context"""
        all_candidates = primary_candidates + pattern_candidates
        
        # Score candidates based on context
        scored_candidates = []
        for candidate in all_candidates:
            score = self.score_candidate(candidate, context)
            scored_candidates.append({
                'candidate': candidate,
                'context_score': score
            })
            
        # Return highest scoring candidate
        if scored_candidates:
            best = max(scored_candidates, key=lambda x: x['context_score'])
            return best['candidate']['intent']
        else:
            return 'unknown'
            
    def score_candidate(self, candidate, context):
        """Score a candidate intent based on context"""
        score = 0.0
        
        # Check if intent is consistent with recent interactions
        if context and hasattr(context, 'recent_intents'):
            if candidate['intent'] in context.recent_intents:
                score += 0.3
                
        # Check if objects in intent match available objects
        if context and hasattr(context, 'available_objects'):
            intent_obj = self.extract_object_from_intent(candidate['intent'])
            for obj in context.available_objects:
                if intent_obj.lower() in obj.name.lower():
                    score += 0.2
                    
        return score
        
    def extract_object_from_intent(self, intent_path):
        """Extract object type from intent path"""
        # Example: manipulation.grasp_object -> object
        parts = intent_path.split('.')
        if len(parts) >= 2:
            action = parts[1]
            # Map action types to object types
            action_to_obj = {
                'grasp_object': 'object',
                'transport_object': 'object',
                'place_object': 'object'
            }
            return action_to_obj.get(action, 'item')
        return 'item'
```

### Handling Complex and Compound Commands

Robots often receive complex commands that require multiple actions:

```cpp
// Handling complex and compound commands
class ComplexCommandProcessor {
public:
    ComplexCommandProcessor() {
        action_sequencer_ = std::make_unique<ActionSequencer>();
        dependency_resolver_ = std::make_unique<DependencyResolver>();
        temporal_planner_ = std::make_unique<TemporalPlanner>();
    }

    ComplexCommandPlan processComplexCommand(const std::string& command) {
        // Parse the complex command into constituent parts
        auto subcommands = parseComplexCommand(command);
        
        // Identify dependencies between subcommands
        auto dependencies = identifyDependencies(subcommands);
        
        // Create a plan with proper sequencing
        auto plan = createSequencedPlan(subcommands, dependencies);
        
        return plan;
    }

private:
    struct SubCommand {
        std::string text;
        std::string intent;
        std::vector<std::string> entities;
        std::string temporal_modifier;  // "then", "after", "while", etc.
    };

    struct ComplexCommandPlan {
        std::vector<RobotAction> actions;
        std::vector<Dependency> action_dependencies;
        TemporalStructure temporal_structure;
        std::string coordination_strategy;  // sequential, parallel, conditional
    };

    std::vector<SubCommand> parseComplexCommand(const std::string& command) {
        std::vector<SubCommand> subcommands;
        
        // Split command by coordinating conjunctions
        std::vector<std::string> conjuctions = {"and", "then", "after that", "next", "finally"};
        
        // For now, split on "and" and "then" as simple example
        std::vector<std::string> parts = splitCommandOnConnectives(command);
        
        for (const auto& part : parts) {
            SubCommand subcmd;
            subcmd.text = part;
            
            // Classify intent for subcommand
            subcmd.intent = classifyIntent(part);
            
            // Extract entities
            subcmd.entities = extractEntities(part);
            
            // Determine temporal relationship
            subcmd.temporal_modifier = determineTemporalModifier(part, command);
            
            subcommands.push_back(subcmd);
        }
        
        return subcommands;
    }
    
    std::vector<std::string> splitCommandOnConnectives(const std::string& command) {
        // Simple splitting on common connectives
        std::vector<std::string> connectives = {" and ", " then ", " after ", " next "};
        
        std::string working_cmd = command;
        std::vector<std::string> parts;
        
        for (const auto& conn : connectives) {
            size_t pos = 0;
            while ((pos = working_cmd.find(conn)) != std::string::npos) {
                std::string part = working_cmd.substr(0, pos);
                parts.push_back(trim(part));
                
                // Move past the connective
                working_cmd = working_cmd.substr(pos + conn.length());
            }
        }
        
        // Add the remaining part
        if (!working_cmd.empty()) {
            parts.push_back(trim(working_cmd));
        }
        
        return parts;
    }
    
    std::string classifyIntent(const std::string& subcommand) {
        // Use intent classifier for each subcommand
        // This would call the appropriate intent classification method
        HierarchicalIntentClassifier classifier;
        auto result = classifier.classifyWithConfidence(subcommand);
        return result.primary;
    }
    
    std::vector<std::string> extractEntities(const std::string& subcommand) {
        // Extract entities from subcommand
        std::vector<std::string> entities;
        
        // Simple approach: extract noun phrases
        // In practice, this would use more sophisticated NER
        std::regex noun_pattern(R"(\b(a|an|the)\s+\w+\b)");
        std::sregex_iterator iter(subcommand.begin(), subcommand.end(), noun_pattern);
        std::sregex_iterator end;
        
        for (; iter != end; ++iter) {
            entities.push_back(iter->str());
        }
        
        return entities;
    }
    
    std::string determineTemporalModifier(const std::string& subcommand, 
                                        const std::string& original_command) {
        // Determine how this subcommand relates temporally to others
        if (original_command.find(subcommand) == 0) {
            return "first";
        } else if (original_command.find("then " + subcommand) != std::string::npos ||
                   original_command.find(subcommand + " then") != std::string::npos) {
            return "subsequent";
        } else {
            return "independent";
        }
    }
    
    std::vector<Dependency> identifyDependencies(const std::vector<SubCommand>& subcommands) {
        std::vector<Dependency> dependencies;
        
        // Identify dependencies between subcommands
        // e.g., "grasp object and place it" - place depends on grasp
        for (size_t i = 1; i < subcommands.size(); ++i) {
            if (hasDependency(subcommands[i-1], subcommands[i])) {
                dependencies.push_back({
                    .from = i-1,
                    .to = i,
                    .type = DependencyType::SEQUENTIAL
                });
            }
        }
        
        return dependencies;
    }
    
    bool hasDependency(const SubCommand& cmd1, const SubCommand& cmd2) {
        // Check if cmd2 depends on cmd1
        // e.g., if cmd2 refers to an object introduced in cmd1
        for (const auto& entity : cmd1.entities) {
            std::string lower_entity = entity;
            std::transform(lower_entity.begin(), lower_entity.end(), 
                          lower_entity.begin(), ::tolower);
            
            std::string lower_cmd2 = cmd2.text;
            std::transform(lower_cmd2.begin(), lower_cmd2.end(), 
                          lower_cmd2.begin(), ::tolower);
            
            // Check for pronouns or references to the entity
            if (lower_cmd2.find("it") != std::string::npos ||
                lower_cmd2.find("that") != std::string::npos ||
                lower_cmd2.find(lower_entity) != std::string::npos) {
                return true;
            }
        }
        
        return false;
    }
    
    ComplexCommandPlan createSequencedPlan(const std::vector<SubCommand>& subcommands,
                                          const std::vector<Dependency>& dependencies) {
        ComplexCommandPlan plan;
        
        // Convert subcommands to robot actions
        for (const auto& subcmd : subcommands) {
            RobotAction action = convertToRobotAction(subcmd);
            plan.actions.push_back(action);
        }
        
        // Add dependencies
        plan.action_dependencies = dependencies;
        
        // Determine temporal structure
        plan.temporal_structure = determineTemporalStructure(dependencies, subcommands.size());
        
        // Determine coordination strategy
        plan.coordination_strategy = determineCoordinationStrategy(dependencies);
        
        return plan;
    }
    
    RobotAction convertToRobotAction(const SubCommand& subcmd) {
        // Convert subcommand to executable robot action
        RobotAction action;
        action.intent = subcmd.intent;
        action.entities = subcmd.entities;
        action.temporal_modifier = subcmd.temporal_modifier;
        
        return action;
    }
    
    TemporalStructure determineTemporalStructure(
        const std::vector<Dependency>& dependencies, size_t total_actions) {
        
        if (dependencies.empty()) {
            return TemporalStructure::PARALLEL;
        } else {
            return TemporalStructure::SEQUENTIAL;
        }
    }
    
    std::string determineCoordinationStrategy(const std::vector<Dependency>& dependencies) {
        // Determine how to coordinate multiple actions
        if (dependencies.empty()) {
            return "parallel";
        } else {
            return "sequential";
        }
    }
    
    std::string trim(const std::string& str) {
        size_t start = str.find_first_not_of(" \t");
        if (start == std::string::npos) return "";
        
        size_t end = str.find_last_not_of(" \t");
        return str.substr(start, end - start + 1);
    }
    
    std::unique_ptr<ActionSequencer> action_sequencer_;
    std::unique_ptr<DependencyResolver> dependency_resolver_;
    std::unique_ptr<TemporalPlanner> temporal_planner_;
};
```

## Semantic Parsing for Robot Commands

### Compositional Semantic Parsing

Semantic parsing converts natural language into structured logical forms that robots can execute:

```python
class CompositionalSemanticParser:
    def __init__(self):
        self.lexicon = self.build_robot_lexicon()
        self.composition_rules = self.define_composition_rules()
        self.executor = RobotCommandExecutor()
        
    def build_robot_lexicon(self):
        """Build a lexicon mapping words to semantic representations"""
        return {
            # Actions
            'grasp': {'type': 'action', 'value': 'grasp', 'args': ['entity']},
            'take': {'type': 'action', 'value': 'grasp', 'args': ['entity']},
            'pick up': {'type': 'action', 'value': 'grasp', 'args': ['entity']},
            'place': {'type': 'action', 'value': 'place', 'args': ['entity', 'location']},
            'put': {'type': 'action', 'value': 'place', 'args': ['entity', 'location']},
            'go': {'type': 'action', 'value': 'navigate', 'args': ['location']},
            'move': {'type': 'action', 'value': 'navigate', 'args': ['location']},
            
            # Entities
            'the': {'type': 'determiner', 'value': 'definite'},
            'a': {'type': 'determiner', 'value': 'indefinite'},
            'red': {'type': 'property', 'value': 'red', 'category': 'color'},
            'blue': {'type': 'property', 'value': 'blue', 'category': 'color'},
            'cup': {'type': 'object', 'value': 'cup', 'category': 'container'},
            'book': {'type': 'object', 'value': 'book', 'category': 'stationery'},
            'table': {'type': 'object', 'value': 'table', 'category': 'furniture'},
            'kitchen': {'type': 'location', 'value': 'kitchen', 'category': 'room'},
            
            # Spatial relations
            'on': {'type': 'spatial', 'value': 'on_top_of', 'args': ['entity', 'entity']},
            'in': {'type': 'spatial', 'value': 'inside', 'args': ['entity', 'entity']},
            'under': {'type': 'spatial', 'value': 'under', 'args': ['entity', 'entity']},
            'next to': {'type': 'spatial', 'value': 'adjacent_to', 'args': ['entity', 'entity']},
            
            # Directions
            'left': {'type': 'direction', 'value': 'left', 'reference': 'egocentric'},
            'right': {'type': 'direction', 'value': 'right', 'reference': 'egocentric'},
            'front': {'type': 'direction', 'value': 'front', 'reference': 'egocentric'},
            'back': {'type': 'direction', 'value': 'back', 'reference': 'egocentric'},
        }
    
    def define_composition_rules(self):
        """Define rules for composing meanings"""
        return {
            # Action + Entity -> Action with Entity
            ('action', 'entity'): self.compose_action_entity,
            
            # Determiner + Adjective + Noun -> Entity
            ('determiner', 'property', 'object'): self.compose_det_adj_noun,
            
            # Action + Entity + Location -> Complex Action
            ('action', 'entity', 'location'): self.compose_action_entity_location,
            
            # Preposition + Noun Phrase -> Spatial Relation
            ('spatial', 'entity'): self.compose_preposition_entity,
        }
    
    def parse_command(self, command):
        """Parse a command into a semantic representation"""
        # Tokenize the command
        tokens = self.tokenize_command(command)
        
        # Get semantic representations for each token
        meanings = [self.lookup_meaning(token) for token in tokens]
        
        # Compose meanings according to grammar rules
        composition_stack = []
        for meaning in meanings:
            composition_stack.append(meaning)
            # Try to apply composition rules
            self.apply_composition_rules(composition_stack)
        
        # The final meaning should be executable
        return self.execute_parse_result(composition_stack[0])
    
    def tokenize_command(self, command):
        """Convert command to tokens"""
        # Handle multi-word units first
        multi_word_units = [
            'pick up', 'next to', 'on top of', 'in front of'
        ]
        
        normalized = command.lower()
        
        # Replace multi-word units with single tokens
        tokenized = normalized
        for multi_word in multi_word_units:
            tokenized = tokenized.replace(multi_word, multi_word.replace(' ', '_'))
        
        return tokenized.split()
    
    def lookup_meaning(self, token):
        """Lookup the meaning for a token"""
        if token in self.lexicon:
            return self.lexicon[token]
        else:
            # Handle unknown words - might be object names or locations
            return {
                'type': 'entity',
                'value': token,
                'category': 'unknown'
            }
    
    def apply_composition_rules(self, stack):
        """Apply composition rules to the stack of meanings"""
        # Try to find applicable rules based on the top items of the stack
        if len(stack) >= 2:
            top_types = tuple(item['type'] for item in stack[-2:])
            if top_types in self.composition_rules:
                rule = self.composition_rules[top_types]
                result = rule(stack[-2], stack[-1])
                # Replace the two items with the result
                stack.pop()  # Remove last item
                stack.pop()  # Remove second-to-last item
                stack.append(result)
    
    def compose_action_entity(self, action, entity):
        """Compose an action with an entity"""
        return {
            'type': 'robot_action',
            'action': action['value'],
            'arguments': {
                'target': entity
            }
        }
    
    def compose_action_entity_location(self, action, entity, location):
        """Compose action with entity and location"""
        return {
            'type': 'robot_action',
            'action': action['value'],
            'arguments': {
                'target': entity,
                'destination': location
            }
        }
    
    def execute_parse_result(self, parse_result):
        """Execute the parsed command"""
        if parse_result['type'] == 'robot_action':
            return self.executor.execute_action(
                parse_result['action'],
                parse_result['arguments']
            )
        else:
            raise ValueError(f"Cannot execute parse result of type {parse_result['type']}")

# Example usage:
# parser = CompositionalSemanticParser()
# result = parser.parse_command("grasp the red cup on the table")
```

### Knowledge-Grounded Semantic Parsing

Advanced semantic parsing incorporates world knowledge:

```cpp
// Knowledge-grounded semantic parsing for robotics
class KnowledgeGroundedParser {
public:
    KnowledgeGroundedParser() {
        initializeKnowledgeBase();
        initializeSemanticParser();
        initializeGroundingModule();
    }

    RobotCommand parseWithWorldKnowledge(const std::string& command,
                                        const WorldState& world_state) {
        // Parse the command syntactically and semantically
        auto semantic_representation = semantic_parser_.parse(command);
        
        // Ground the representation in the current world state
        auto grounded_representation = grounding_module_.ground(
            semantic_representation, world_state
        );
        
        // Use knowledge base to refine the interpretation
        auto refined_command = knowledge_refiner_.refine(
            grounded_representation, world_state
        );
        
        return refined_command;
    }

private:
    struct SemanticRepresentation {
        std::string action_type;
        std::vector<Entity> entities;
        std::vector<Relation> spatial_relations;
        std::vector<Constraint> execution_constraints;
    };

    struct GroundedRepresentation {
        SemanticRepresentation semantic;
        std::map<std::string, ObjectInWorld> entity_groundings;
        std::vector<GroundedRelation> grounded_relations;
        ExecutionContext execution_context;
    };

    struct RobotCommand {
        std::string action_type;
        ObjectInWorld target_object;
        Location destination;
        std::vector<Constraint> constraints;
        ExecutionParameters parameters;
    };

    struct KnowledgeRefiner {
        // Refine commands based on:
        // - Physical constraints (object sizes, weights, affordances)
        // - Functional relationships (objects that typically go together)
        // - Pragmatic knowledge (social norms, typical usage patterns)
        
        RobotCommand refine(const GroundedRepresentation& grounded_repr,
                           const WorldState& world_state) {
            RobotCommand command;
            
            // Apply physical constraints
            command = applyPhysicalConstraints(grounded_repr, world_state);
            
            // Apply functional knowledge
            command = applyFunctionalKnowledge(command, world_state);
            
            // Apply pragmatic knowledge (social norms, etc.)
            command = applyPragmaticKnowledge(command, world_state);
            
            return command;
        }

    private:
        RobotCommand applyPhysicalConstraints(const GroundedRepresentation& repr,
                                            const WorldState& world_state) {
            RobotCommand cmd = initializeFromGrounded(repr);
            
            // Verify physical feasibility
            if (repr.semantic.action_type == "grasp") {
                // Check if object is graspable based on properties
                auto obj = repr.entity_groundings.at("target");
                if (obj.properties.mass > robot_capability_.max_grasp_weight) {
                    throw PhysicalConstraintException(
                        "Object too heavy to grasp"
                    );
                }
                if (obj.properties.size.volume > robot_capability_.max_grasp_size) {
                    throw PhysicalConstraintException(
                        "Object too large to grasp"
                    );
                }
            }
            
            return cmd;
        }
        
        RobotCommand applyFunctionalKnowledge(const RobotCommand& cmd,
                                           const WorldState& world_state) {
            RobotCommand refined = cmd;
            
            // Apply functional knowledge
            // e.g., if asked to "make coffee", understand it involves multiple steps
            if (cmd.action_type == "make" && cmd.target_object.category == "coffee") {
                // Expand to sequence: go to kitchen, find coffee maker, 
                // add water, add coffee beans, activate coffee maker
                refined = expandToFunctionalSequence("make_coffee");
            }
            
            // Apply object co-occurrence knowledge
            // e.g., if asked for "wine", likely need "wine glass" too
            if (cmd.target_object.category == "wine") {
                auto nearby_glasses = world_state.getObjectByCategory("wine_glass");
                if (!nearby_glasses.empty()) {
                    refined.associated_objects.push_back(nearby_glasses[0]);
                }
            }
            
            return refined;
        }
        
        RobotCommand applyPragmaticKnowledge(const RobotCommand& cmd,
                                           const WorldState& world_state) {
            RobotCommand refined = cmd;
            
            // Apply social/pragmatic knowledge
            // e.g., when passing objects, consider safety and convenience
            if (cmd.action_type == "deliver") {
                // Choose approach direction that's convenient for the recipient
                refined.approach_direction = 
                    determineConvenientApproach(cmd.destination, world_state);
            }
            
            return refined;
        }
        
        RobotCommand initializeFromGrounded(const GroundedRepresentation& repr) {
            RobotCommand cmd;
            cmd.action_type = repr.semantic.action_type;
            
            if (!repr.entity_groundings.empty()) {
                cmd.target_object = repr.entity_groundings.begin()->second;
            }
            
            return cmd;
        }
        
        RobotCommand expandToFunctionalSequence(const std::string& function) {
            // Return a sequence of primitive robot commands
            // that achieve a complex function
            return RobotCommand{}; // Placeholder
        }
        
        std::string determineConvenientApproach(const Location& destination,
                                               const WorldState& world_state) {
            // Determine the most convenient direction to approach a location
            // considering human posture, safety, etc.
            return "front"; // Placeholder
        }
        
        RobotCapabilities robot_capability_;
    };

    void initializeKnowledgeBase();
    void initializeSemanticParser();
    void initializeGroundingModule();
    
    SemanticParser semantic_parser_;
    GroundingModule grounding_module_;
    KnowledgeRefiner knowledge_refiner_;
    KnowledgeBase knowledge_base_;
};
```

## Context-Aware Language Processing

### Maintaining Discourse Context

Context-aware processing maintains information across multiple interactions:

```python
class ContextAwareLanguageProcessor:
    def __init__(self):
        self.dialogue_context = DialogueContext()
        self.spatial_context = SpatialContext()
        self.task_context = TaskContext()
        self.user_context = UserContext()
        
    def process_command_with_context(self, command, environment_state):
        """Process command using all available context"""
        # Update context with current environment
        self.spatial_context.update_from_environment(environment_state)
        
        # Parse command with context grounding
        parsed_command = self.parse_with_context(command)
        
        # Resolve references using context
        resolved_command = self.resolve_with_context(
            parsed_command, environment_state
        )
        
        # Update dialogue context for future interactions
        self.dialogue_context.update_with_command(command, resolved_command)
        
        return resolved_command
        
    def parse_with_context(self, command):
        """Parse command considering discourse context"""
        # Use context to disambiguate references
        resolved_command = self.disambiguate_with_context(command)
        
        # Extract entities considering context
        entities = self.extract_entities_with_context(
            resolved_command, self.dialogue_context
        )
        
        # Determine action considering task context
        action = self.determine_action_with_context(
            resolved_command, self.task_context
        )
        
        return {
            'command': resolved_command,
            'entities': entities,
            'action': action,
            'spatial_ref': self.extract_spatial_ref(command)
        }
    
    def disambiguate_with_context(self, command):
        """Disambiguate command using discourse context"""
        # Handle anaphoric references
        command = self.resolve_pronouns(command)
        
        # Handle ellipsis
        command = self.expand_ellipsis(command)
        
        # Use spatial context for location references
        command = self.resolve_spatial_references(command)
        
        return command
        
    def resolve_pronouns(self, command):
        """Resolve pronouns using discourse context"""
        # Identify pronouns in the command
        pronouns = ['it', 'they', 'them', 'this', 'that']
        
        for pronoun in pronouns:
            if pronoun in command.lower():
                # Find antecedent in recent context
                antecedent = self.find_antecedent(pronoun)
                if antecedent:
                    command = command.replace(pronoun, antecedent)
                    
        return command
        
    def find_antecedent(self, pronoun):
        """Find the most likely antecedent for a pronoun"""
        # Look in recent discourse context
        for item in reversed(self.dialogue_context.recent_entities):
            if self.is_suitable_antecedent(pronoun, item):
                return item
                
        # If no entity found, look for salient objects in spatial context
        if hasattr(self.spatial_context, 'salient_objects'):
            if pronoun in ['it', 'this', 'that']:
                return self.spatial_context.salient_objects[0] if self.spatial_context.salient_objects else None
                
        return None
        
    def is_suitable_antecedent(self, pronoun, entity):
        """Check if entity is a suitable antecedent for pronoun"""
        if pronoun in ['it', 'this', 'that']:
            # Any entity can be antecedent
            return True
        elif pronoun in ['they', 'them']:
            # Should be plural or collection
            return hasattr(entity, 'is_plural') and entity.is_plural
            
        return False
        
    def extract_entities_with_context(self, command, context):
        """Extract entities using context to improve recognition"""
        # Initial extraction
        entities = self.extract_entities(command)
        
        # Use context to disambiguate overlapping entities
        disambiguated_entities = self.disambiguate_entities_with_context(
            entities, context
        )
        
        # Use spatial context to ground entities in the environment
        grounded_entities = self.ground_entities_in_environment(
            disambiguated_entities, self.spatial_context
        )
        
        return grounded_entities
        
    def disambiguate_entities_with_context(self, entities, context):
        """Disambiguate entities using context"""
        for i, entity in enumerate(entities):
            if entity.name == 'ambiguous':
                # Use context to disambiguate
                disambiguated = self.disambiguate_entity(entity, context)
                if disambiguated:
                    entities[i] = disambiguated
                    
        return entities
        
    def update_context_after_execution(self, command, result):
        """Update context based on command execution result"""
        # Update dialogue context
        self.dialogue_context.update_with_result(command, result)
        
        # Update task context
        self.task_context.update_with_result(command, result)
        
        # Update spatial context based on changes in environment
        self.spatial_context.update_from_result(result)

class DialogueContext:
    def __init__(self):
        self.recent_entities = []
        self.current_task = None
        self.user_intention = None
        self.dialogue_history = []
        self.referring_expressions = {}
        
    def update_with_command(self, command, parsed_result):
        """Update context with information from new command"""
        # Add entities mentioned in command
        if 'entities' in parsed_result:
            for entity in parsed_result['entities']:
                self.recent_entities.append(entity)
                # Keep only recent entities
                if len(self.recent_entities) > 10:
                    self.recent_entities.pop(0)
                    
        # Update dialogue history
        self.dialogue_history.append({
            'command': command,
            'parsed': parsed_result,
            'timestamp': time.time()
        })
        
        # Track current task if mentioned
        if 'task' in parsed_result:
            self.current_task = parsed_result['task']
            
    def update_with_result(self, command, result):
        """Update context with execution result"""
        # Update referring expressions if an object was uniquely identified
        if result.get('target_object'):
            obj = result['target_object']
            self.referring_expressions[command] = obj
            
    def get_salient_entities(self, limit=3):
        """Get most salient entities in current context"""
        return self.recent_entities[-limit:]

class SpatialContext:
    def __init__(self):
        self.objects = {}
        self.locations = {}
        self.spatial_relations = {}
        self.salient_objects = []
        
    def update_from_environment(self, env_state):
        """Update spatial context from environment state"""
        self.objects = env_state.get('objects', {})
        self.locations = env_state.get('locations', {})
        
        # Update spatial relations
        self.spatial_relations = self.compute_spatial_relations()
        
        # Identify salient objects (closest, most interacted with, etc.)
        self.salient_objects = self.identify_salient_objects()
        
    def compute_spatial_relations(self):
        """Compute spatial relations between objects"""
        relations = {}
        
        obj_list = list(self.objects.values())
        for i, obj1 in enumerate(obj_list):
            for j, obj2 in enumerate(obj_list):
                if i != j:
                    relation = self.compute_object_relationship(obj1, obj2)
                    relations[f"{obj1.id}-{obj2.id}"] = relation
                    
        return relations
        
    def ground_entities_in_environment(self, entities, env_state):
        """Ground entity references in the environment"""
        grounded_entities = []
        
        for entity in entities:
            grounded = self.ground_single_entity(entity, env_state)
            grounded_entities.append(grounded)
            
        return grounded_entities
        
    def ground_single_entity(self, entity, env_state):
        """Ground a single entity in the environment"""
        # Find matching object in environment
        candidates = self.find_matching_objects(entity, env_state)
        
        if len(candidates) == 1:
            # Unique match
            entity.grounded_reference = candidates[0]
        elif len(candidates) > 1:
            # Multiple matches - more context needed
            entity.ambiguous_reference = candidates
        else:
            # No matches - may need clarification
            entity.no_reference_found = True
            
        return entity
```

## Multilingual Support

### Handling Multiple Languages

Robots working in diverse environments need multilingual capabilities:

```python
class MultilingualLanguageProcessor:
    def __init__(self):
        self.language_detectors = {}
        self.translators = {}
        self.mono_language_processors = {}
        self.initialize_languages(['en', 'es', 'fr', 'de', 'ur'])  # English, Spanish, French, German, Urdu
        
    def initialize_languages(self, languages):
        """Initialize support for specified languages"""
        for lang in languages:
            # Initialize language detection
            self.language_detectors[lang] = LanguageDetector(lang)
            
            # Initialize translation to English for processing
            self.translators[lang] = TranslationModule(f"{lang}_to_en")
            
            # Initialize monolingual processor for this language
            self.mono_language_processors[lang] = MonoLingualProcessor(lang)
    
    def process_command_multilingual(self, command, environment_context):
        """Process command in any supported language"""
        # Detect input language
        detected_language = self.detect_language(command)
        
        # If not English, translate to English for processing
        if detected_language != 'en':
            english_command = self.translators[detected_language].translate(command)
        else:
            english_command = command
            
        # Process the English version
        result = self.mono_language_processors['en'].process_command(
            english_command, environment_context
        )
        
        # If needed, translate result back to original language
        if detected_language != 'en':
            result = self.translate_result(result, detected_language)
            
        return result
        
    def detect_language(self, text):
        """Detect the language of input text"""
        # This would use a language detection model
        # For simplicity, we'll check basic patterns
        urdu_chars = set(['\u0600-\u06FF'])  # Arabic/Urdu script range
        
        # Check if text contains Urdu script
        if any('\u0600' <= c <= '\u06FF' for c in text):
            return 'ur'
        
        # For other languages, use a more sophisticated approach
        # Here we'll use a simple word-based detection
        common_words = {
            'en': ['the', 'and', 'or', 'what', 'where', 'how'],
            'es': ['el', 'la', 'y', 'o', 'qué', 'dónde', 'cómo'],
            'fr': ['le', 'la', 'et', 'ou', 'quand', 'comment', 'où'],
            'de': ['der', 'die', 'und', 'oder', 'was', 'wo', 'wie']
        }
        
        text_lower = text.lower()
        scores = {}
        
        for lang, words in common_words.items():
            score = sum(1 for word in words if f" {word} " in f" {text_lower} ")
            scores[lang] = score
            
        # Return language with highest score
        if scores:
            return max(scores, key=scores.get)
        else:
            return 'en'  # Default to English
            
    def translate_result(self, result, target_language):
        """Translate result to target language"""
        if target_language == 'en':
            return result
            
        # Translate relevant parts of the result
        translated_result = result.copy()
        
        if 'response_text' in result:
            translated_result['response_text'] = self.translators[f"en_to_{target_language}"].translate(
                result['response_text']
            )
            
        return translated_result

class MonoLingualProcessor:
    def __init__(self, language):
        self.language = language
        self.intent_classifier = IntentClassifier(lang=language)
        self.spatial_processor = SpatialLanguageProcessor(lang=language)
        self.context_handler = ContextAwareProcessor(lang=language)
        
    def process_command(self, command, environment_context):
        """Process command in the specific language"""
        # Classify intent
        intent = self.intent_classifier.classify(command)
        
        # Process spatial language
        spatial_info = self.spatial_processor.process_spatial_command(
            command, environment_context
        )
        
        # Handle context
        contextual_result = self.context_handler.process_with_context(
            command, intent, spatial_info, environment_context
        )
        
        return contextual_result

# Urdu-specific processing
class UrduLanguageProcessor:
    def __init__(self):
        self.nastaliq_font_support = True
        self.urdu_patterns = self.load_urdu_patterns()
        self.urdu_semantic_parser = self.setup_urdu_semantics()
        
    def load_urdu_patterns(self):
        """Load Urdu-specific language patterns"""
        return {
            'navigation': ['وہاں جاؤ', 'وہاں چلیں', 'وہ کمرہ', 'وہ جگہ'],
            'manipulation': ['اسے اٹھاؤ', 'اسے لے لو', 'اسے چھوڑو', 'اسے رکھو'],
            'social': ['ہیلو', 'السلام علیکم', 'خیریت ہے', 'آداب'],
            'question': ['کیا', 'کہاں', 'کیوں', 'کیسے', 'کب']
        }
    
    def preprocess_urdu(self, text):
        """Preprocess Urdu text for processing"""
        # Normalize Urdu text
        normalized = self.normalize_urdu_text(text)
        
        # Handle RTL text issues if necessary
        # Tokenize according to Urdu rules
        tokens = self.tokenize_urdu_text(normalized)
        
        return tokens
        
    def normalize_urdu_text(self, text):
        """Normalize Urdu text"""
        # Apply Urdu-specific normalization
        # Remove extra spaces, normalize characters, etc.
        return text.strip()
        
    def tokenize_urdu_text(self, text):
        """Tokenize Urdu text"""
        # Urdu-specific tokenization
        # For now, simple whitespace tokenization with punctuation handling
        import re
        # Split on spaces and common punctuation but preserve Urdu characters
        tokens = re.split(r'[\s،۔!?؟\-\(\)]+', text)
        return [token for token in tokens if token.strip()]
        
    def parse_urdu_command(self, command, environment):
        """Parse Urdu command with spatial and contextual understanding"""
        # Preprocess command
        tokens = self.tokenize_urdu_text(command)
        
        # Identify intent using Urdu patterns
        intent = self.identify_urdu_intent(command)
        
        # Extract entities considering Urdu grammar
        entities = self.extract_urdu_entities(tokens, environment)
        
        # Handle spatial references in Urdu context
        spatial_ref = self.handle_urdu_spatial_ref(command, environment)
        
        return {
            'intent': intent,
            'entities': entities,
            'spatial_ref': spatial_ref,
            'original_text': command
        }
        
    def identify_urdu_intent(self, command):
        """Identify intent in Urdu command"""
        for intent, patterns in self.urdu_patterns.items():
            for pattern in patterns:
                if pattern in command:
                    return intent
        return 'unknown'
        
    def extract_urdu_entities(self, tokens, environment):
        """Extract entities from Urdu tokens"""
        entities = []
        
        # Look for common Urdu object names
        urdu_object_names = {
            'کتاب': 'book',
            'کپ': 'cup', 
            'پلیٹ': 'plate',
            'کرسی': 'chair',
            'میز': 'table',
            'در': 'door',
            'کھڑکی': 'window'
        }
        
        for token in tokens:
            if token in urdu_object_names:
                # Find this object in the environment
                obj_in_env = self.find_object_in_environment(
                    urdu_object_names[token], environment
                )
                if obj_in_env:
                    entities.append({
                        'urdu_name': token,
                        'english_name': urdu_object_names[token],
                        'environment_object': obj_in_env
                    })
                    
        return entities
        
    def find_object_in_environment(self, object_name, environment):
        """Find object in the environment by name"""
        for obj in environment.get('objects', []):
            if obj.get('name', '').lower() == object_name.lower():
                return obj
        return None

# Combined multilingual processor with Urdu support
class AdvancedMultilingualProcessor:
    def __init__(self):
        self.monolingual_processors = {}
        self.urdu_processor = UrduLanguageProcessor()
        self.translation_module = TranslationModule()
        self.initialize_processors(['en', 'ur'])
        
    def initialize_processors(self, languages):
        """Initialize processors for each language"""
        for lang in languages:
            if lang == 'ur':
                self.monolingual_processors[lang] = self.urdu_processor
            else:
                self.monolingual_processors[lang] = MonoLingualProcessor(lang)
                
    def process_command(self, command, environment_context, preferred_language=None):
        """Process command in the appropriate language"""
        # If language is specified, use that processor
        if preferred_language and preferred_language in self.monolingual_processors:
            return self.monolingual_processors[preferred_language].process_command(
                command, environment_context
            )
        else:
            # Otherwise, detect language first
            detected_lang = self.detect_language(command)
            return self.monolingual_processors[detected_lang].process_command(
                command, environment_context
            )
            
    def detect_language(self, text):
        """Detect language of text"""
        # Check for Urdu script first
        if any('\u0600' <= c <= '\u06FF' for c in text):
            return 'ur'
        
        # For other languages, use basic detection
        # This is a simplified implementation
        return 'en'  # Default to English
```

## Learning from Interaction

### Online Learning for Language Understanding

Robots should improve their language understanding through continued interaction:

```python
class OnlineLanguageLearner:
    def __init__(self):
        self.language_model = NeuralLanguageModel()
        self.user_interaction_model = UserInteractionModel()
        self.feedback_processor = FeedbackProcessor()
        self.experience_replay_buffer = ExperienceReplayBuffer()
        
    def learn_from_interaction(self, user_input, robot_response, 
                              execution_result, user_feedback):
        """Learn from a single interaction episode"""
        # Process user feedback
        feedback_analysis = self.feedback_processor.analyze(user_feedback)
        
        # Update language model based on outcome
        self.update_language_model(
            user_input, robot_response, execution_result, feedback_analysis
        )
        
        # Update user interaction model
        self.update_user_model(user_input, user_feedback)
        
        # Store experience for replay
        experience = {
            'input': user_input,
            'response': robot_response,
            'result': execution_result,
            'feedback': user_feedback,
            'feedback_analysis': feedback_analysis,
            'timestamp': time.time()
        }
        self.experience_replay_buffer.add(experience)
        
        # Occasionally update model with replayed experiences
        if len(self.experience_replay_buffer) > 100:
            replay_batch = self.experience_replay_buffer.sample(32)
            self.update_with_replay(replay_batch)
    
    def update_language_model(self, user_input, robot_response, 
                             execution_result, feedback_analysis):
        """Update the language understanding model"""
        # Calculate reward signal from execution and feedback
        reward = self.calculate_reward(execution_result, feedback_analysis)
        
        # Prepare training data
        training_data = {
            'input': user_input,
            'target_action': robot_response.get('intended_action', ''),
            'reward': reward,
            'context': robot_response.get('context', {})
        }
        
        # Update the model
        self.language_model.update(training_data)
    
    def calculate_reward(self, execution_result, feedback_analysis):
        """Calculate reward from execution result and user feedback"""
        reward = 0.0
        
        # Execution success contributes to reward
        if execution_result.get('success', False):
            reward += 1.0
        else:
            reward -= 0.5
            
        # Positive feedback contributes to reward
        if feedback_analysis.get('positivity', 0) > 0:
            reward += feedback_analysis['positivity'] * 0.5
            
        # Negative feedback reduces reward
        if feedback_analysis.get('negativity', 0) > 0:
            reward -= feedback_analysis['negativity'] * 0.3
            
        # Task completion contributes to reward
        if feedback_analysis.get('task_completed', False):
            reward += 1.0
            
        return max(-1.0, min(1.0, reward))  # Clamp to [-1, 1]
    
    def update_user_model(self, user_input, user_feedback):
        """Update model of user preferences and communication style"""
        # Analyze user's communication style
        style_features = self.analyze_communication_style(user_input)
        
        # Update user model
        self.user_interaction_model.update_user_profile(
            user_input, user_feedback, style_features
        )
    
    def adapt_to_user(self, user_id):
        """Adapt language processing to specific user"""
        user_profile = self.user_interaction_model.get_user_profile(user_id)
        
        if user_profile:
            # Adjust language processing parameters based on user
            self.language_model.adjust_to_user(user_profile)
            
            # Use user-specific translation/model parameters
            return self.create_user_adapted_processor(user_profile)
        else:
            return self  # Return default processor if no profile
            
    def analyze_communication_style(self, text):
        """Analyze user's communication style"""
        features = {
            'formality': self.assess_formality(text),
            'directness': self.assess_directness(text),
            'verbosity': self.assess_verbosity(text),
            'preference_indicators': self.extract_preference_indicators(text)
        }
        
        return features
    
    def assess_formality(self, text):
        """Assess formality level of text"""
        formal_indicators = ['please', 'thank you', 'sir', 'madam', 'would you', 'could you']
        informal_indicators = ['hey', 'hi', 'dude', 'gimme', 'wanna']
        
        formal_count = sum(1 for indicator in formal_indicators if indicator in text.lower())
        informal_count = sum(1 for indicator in informal_indicators if indicator in text.lower())
        
        total_indicators = formal_count + informal_count
        if total_indicators == 0:
            return 0.5  # Neutral
            
        return formal_count / total_indicators
    
    def assess_directness(self, text):
        """Assess directness of communication"""
        direct_indicators = ['go to', 'pick up', 'bring me', 'open', 'close']
        indirect_indicators = ['can you', 'would you', 'could you please', 'if you don\'t mind']
        
        direct_count = sum(1 for indicator in direct_indicators if indicator in text.lower())
        indirect_count = sum(1 for indicator in indirect_indicators if indicator in text.lower())
        
        total_indicators = direct_count + indirect_count
        if total_indicators == 0:
            return 0.5  # Neutral
            
        return direct_count / total_indicators
    
    def create_user_adapted_processor(self, user_profile):
        """Create a language processor adapted to the user"""
        class AdaptedProcessor:
            def __init__(self, base_processor, user_profile):
                self.base_processor = base_processor
                self.user_profile = user_profile
                
            def process_command(self, command, context):
                # Adapt processing based on user profile
                if self.user_profile.get('prefers_formal', False):
                    # Ensure appropriate formal responses
                    pass
                elif self.user_profile.get('prefers_direct', False):
                    # Process direct commands more easily
                    pass
                    
                return self.base_processor.process_command(command, context)
        
        return AdaptedProcessor(self, user_profile)

class FeedbackProcessor:
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.implicit_feedback_detector = ImplicitFeedbackDetector()
        
    def analyze(self, feedback):
        """Analyze user feedback"""
        analysis = {}
        
        # Explicit feedback analysis
        if isinstance(feedback, str):
            sentiment = self.sentiment_analyzer.analyze(feedback)
            analysis.update(sentiment)
            
        # Implicit feedback from behavior
        if 'behavior_indicators' in feedback:
            implicit_feedback = self.implicit_feedback_detector.analyze(
                feedback['behavior_indicators']
            )
            analysis.update(implicit_feedback)
            
        # Task completion assessment
        analysis['task_completed'] = self.assess_task_completion(
            feedback.get('task_outcome', {})
        )
        
        return analysis
        
    def assess_task_completion(self, task_outcome):
        """Assess if the task was completed based on outcome"""
        # Implementation would evaluate task outcome
        return task_outcome.get('completed', False)

class ExperienceReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size
        
    def add(self, experience):
        """Add experience to buffer"""
        self.buffer.append(experience)
        if len(self.buffer) > self.max_size:
            # Remove oldest experiences
            self.buffer = self.buffer[-self.max_size:]
            
    def sample(self, batch_size):
        """Sample experiences from buffer"""
        if len(self.buffer) < batch_size:
            return self.buffer
            
        # Random sampling
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
        
    def __len__(self):
        return len(self.buffer)
```

## Evaluation of Language Systems

### Evaluation Metrics for Robot Language Systems

Evaluating natural language processing systems for robots requires metrics that go beyond traditional NLP tasks:

```python
# Evaluation metrics for robot language systems
ROBOT_LANGUAGE_METRICS = {
    'comprehension_accuracy': {
        'definition': 'Percentage of commands correctly understood and mapped to appropriate actions',
        'calculation': 'TP / (TP + FP) where TP is correct action mapping, FP is incorrect mapping',
        'target': 0.90,
        'importance': 'Critical for correct robot behavior'
    },
    'execution_success_rate': {
        'definition': 'Percentage of commands that result in successful task completion',
        'calculation': 'Successful executions / Total commands',
        'target': 0.85,
        'importance': 'Measures end-to-end system effectiveness'
    },
    'response_time': {
        'definition': 'Time from command receipt to robot action initiation',
        'calculation': 'Average time in seconds',
        'target': 2.0,  # seconds
        'importance': 'Affects user experience in interactive scenarios'
    },
    'robustness_to_noise': {
        'definition': 'Accuracy when commands are presented with background noise',
        'calculation': 'Accuracy with noise / Accuracy without noise',
        'target': 0.85,
        'importance': 'Critical for real-world deployment'
    },
    'spatial_grounding_accuracy': {
        'definition': 'Accuracy of grounding spatial language to specific environmental entities',
        'calculation': 'Correctly grounded references / Total spatial references',
        'target': 0.88,
        'importance': 'Critical for navigation and manipulation tasks'
    },
    'pragmatic_appropriateness': {
        'definition': 'How appropriately the robot's behavior matches social expectations',
        'calculation': 'Human-rated appropriateness score (1-5 scale)',
        'target': 4.0,
        'importance': 'Affects human-robot interaction quality'
    }
}

class RobotLanguageEvaluator:
    def __init__(self):
        self.comprehension_evaluator = ComprehensionEvaluator()
        self.execution_evaluator = ExecutionEvaluator()
        self.response_time_evaluator = ResponseTimeEvaluator()
        self.noise_robustness_evaluator = NoiseRobustnessEvaluator()
        self.spatial_grounding_evaluator = SpatialGroundingEvaluator()
        self.social_appropriateness_evaluator = SocialAppropriatenessEvaluator()
        self.human_evaluator = HumanEvaluator()
        
    def evaluate_language_system(self, system, test_suite):
        """Comprehensive evaluation of a robot language system"""
        results = {}
        
        # Evaluate comprehension accuracy
        results['comprehension_accuracy'] = self.comprehension_evaluator.evaluate(
            system, test_suite.comprehension_tests
        )
        
        # Evaluate execution success
        results['execution_success_rate'] = self.execution_evaluator.evaluate(
            system, test_suite.execution_tests
        )
        
        # Evaluate response time
        results['response_time'] = self.response_time_evaluator.evaluate(
            system, test_suite.timing_tests
        )
        
        # Evaluate noise robustness
        results['robustness_to_noise'] = self.noise_robustness_evaluator.evaluate(
            system, test_suite.noise_tests
        )
        
        # Evaluate spatial grounding
        results['spatial_grounding_accuracy'] = self.spatial_grounding_evaluator.evaluate(
            system, test_suite.spatial_tests
        )
        
        # Evaluate social appropriateness
        results['pragmatic_appropriateness'] = self.social_appropriateness_evaluator.evaluate(
            system, test_suite.social_tests
        )
        
        # Incorporate human evaluation
        human_ratings = self.human_evaluator.evaluate(
            system, test_suite.human_interaction_tests
        )
        results['human_rated_quality'] = human_ratings
        
        # Calculate overall score
        results['overall_score'] = self.calculate_overall_score(results)
        
        # Generate detailed report
        report = self.generate_evaluation_report(results, test_suite)
        
        return results, report
    
    def calculate_overall_score(self, results):
        """Calculate an overall score combining all metrics"""
        # Weighted average based on importance
        weights = {
            'comprehension_accuracy': 0.25,
            'execution_success_rate': 0.30,
            'response_time': 0.10,  # Lower weight since it's a timing metric
            'robustness_to_noise': 0.10,
            'spatial_grounding_accuracy': 0.15,
            'pragmatic_appropriateness': 0.10
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in results and isinstance(results[metric], (int, float)):
                # Normalize scores to be between 0 and 1
                normalized_score = min(1.0, max(0.0, results[metric]))
                weighted_score += normalized_score * weight
                total_weight += weight
                
        if total_weight > 0:
            return weighted_score / total_weight
        else:
            return 0.0
    
    def generate_evaluation_report(self, results, test_suite):
        """Generate detailed evaluation report"""
        report = {
            'summary': {
                'overall_score': results['overall_score'],
                'number_of_tests': len(test_suite.all_tests),
                'evaluation_date': datetime.now().isoformat(),
                'system_version': getattr(test_suite, 'system_version', 'unknown')
            },
            'detailed_results': results,
            'strengths': self.identify_strengths(results),
            'weaknesses': self.identify_weaknesses(results),
            'recommendations': self.generate_recommendations(results),
            'comparison_to_baselines': self.compare_to_baselines(results)
        }
        
        return report
    
    def identify_strengths(self, results):
        """Identify system strengths based on evaluation results"""
        strengths = []
        
        if results.get('execution_success_rate', 0) >= 0.9:
            strengths.append("High task completion success rate")
            
        if results.get('comprehension_accuracy', 0) >= 0.9:
            strengths.append("High command comprehension accuracy")
            
        if results.get('robustness_to_noise', 0) >= 0.9:
            strengths.append("Robust to acoustic noise")
            
        if results.get('spatial_grounding_accuracy', 0) >= 0.9:
            strengths.append("Accurate spatial language grounding")
            
        return strengths
    
    def identify_weaknesses(self, results):
        """Identify system weaknesses based on evaluation results"""
        weaknesses = []
        
        if results.get('execution_success_rate', 1) < 0.7:
            weaknesses.append("Low task completion rate")
            
        if results.get('comprehension_accuracy', 1) < 0.75:
            weaknesses.append("Poor command comprehension")
            
        if results.get('spatial_grounding_accuracy', 1) < 0.75:
            weaknesses.append("Inaccurate spatial reference resolution")
            
        if results.get('response_time', float('inf')) > 5.0:
            weaknesses.append("Slow response times")
            
        return weaknesses
        
    def generate_recommendations(self, results):
        """Generate improvement recommendations based on results"""
        recommendations = []
        
        if results.get('spatial_grounding_accuracy', 1) < 0.8:
            recommendations.append({
                'focus_area': 'Spatial grounding',
                'recommendation': 'Improve object detection and spatial relationship modeling',
                'priority': 'high'
            })
            
        if results.get('comprehension_accuracy', 1) < 0.8:
            recommendations.append({
                'focus_area': 'Language understanding',
                'recommendation': 'Enhance semantic parsing and ambiguity resolution',
                'priority': 'high'
            })
            
        if results.get('execution_success_rate', 1) < 0.8:
            recommendations.append({
                'focus_area': 'Execution planning',
                'recommendation': 'Improve error recovery and action planning',
                'priority': 'high'
            })
            
        return recommendations

class HumanEvaluator:
    def __init__(self):
        self.rating_scales = self.define_rating_scales()
        self.bias_mitigation = BiasMitigationProtocol()
        
    def evaluate(self, system, interaction_tests):
        """Evaluate system through human interaction tests"""
        ratings = {
            'naturalness': [],
            'helpfulness': [],
            'ease_of_use': [],
            'social_appropriateness': [],
            'willingness_to_use_again': []
        }
        
        for test in interaction_tests:
            # Conduct human interaction trial
            trial_result = self.conduct_interaction_trial(system, test)
            
            # Collect ratings
            ratings['naturalness'].append(trial_result.get('naturalness', 3))
            ratings['helpfulness'].append(trial_result.get('helpfulness', 3))
            ratings['ease_of_use'].append(trial_result.get('ease_of_use', 3))
            ratings['social_appropriateness'].append(trial_result.get('social_appropriateness', 3))
            ratings['willingness_to_use_again'].append(trial_result.get('willingness_to_use_again', 3))
        
        # Calculate average ratings
        avg_ratings = {k: np.mean(v) for k, v in ratings.items()}
        
        return avg_ratings
    
    def conduct_interaction_trial(self, system, test_scenario):
        """Conduct a single human interaction trial"""
        # Simulate human-robot interaction
        system_response = system.process_command(
            test_scenario.command, test_scenario.environment_context
        )
        
        # Execute the resulting action (in simulation or real-world)
        execution_result = self.execute_action_safely(
            system_response.action, test_scenario.environment_context
        )
        
        # Collect human ratings
        human_ratings = self.collect_human_ratings(
            test_scenario, system_response, execution_result
        )
        
        return human_ratings
    
    def collect_human_ratings(self, scenario, system_response, execution_result):
        """Collect structured human ratings of system performance"""
        # Present rating scales to human evaluator
        ratings = {}
        
        for aspect, scale in self.rating_scales.items():
            # For automated evaluation, we'll simulate human responses
            # In a real scenario, this would be actual human input
            ratings[aspect] = self.simulate_human_rating(aspect, system_response, execution_result)
            
        return ratings
        
    def simulate_human_rating(self, aspect, system_response, execution_result):
        """Simulate human rating for evaluation purposes"""
        # This would normally be actual human input
        # Here we provide realistic simulated ratings based on performance
        if aspect == 'naturalness':
            return 4.0 if system_response.get('used_appropriate_language', True) else 2.0
        elif aspect == 'helpfulness':
            return 4.5 if execution_result.get('success', False) else 2.5
        elif aspect == 'ease_of_use':
            return 4.0 if system_response.get('response_time', 5) < 3 else 3.0
        elif aspect == 'social_appropriateness':
            return 4.0 if system_response.get('used_polite_language', True) else 2.0
        elif aspect == 'willingness_to_use_again':
            return 4.0 if execution_result.get('success', False) else 2.0
        else:
            return 3.0  # Neutral rating
    
    def define_rating_scales(self):
        """Define rating scales for different aspects"""
        return {
            'naturalness': {
                'description': 'How natural and human-like the interaction felt',
                'scale': '1-5 Likert scale (1=very unnatural, 5=very natural)'
            },
            'helpfulness': {
                'description': 'How helpful the robot was in accomplishing the task',
                'scale': '1-5 Likert scale (1=not helpful, 5=very helpful)'
            },
            'ease_of_use': {
                'description': 'How easy it was to communicate with the robot',
                'scale': '1-5 Likert scale (1=very difficult, 5=very easy)'
            },
            'social_appropriateness': {
                'description': 'How socially appropriate the robot\'s behavior was',
                'scale': '1-5 Likert scale (1=very inappropriate, 5=very appropriate)'
            },
            'willingness_to_use_again': {
                'description': 'Likelihood of using this robot again',
                'scale': '1-5 Likert scale (1=definitely not, 5=definitely yes)'
            }
        }
```

## Future Directions

### Emerging Trends in Robot Language Processing

The field of natural language processing for robot control continues to evolve rapidly:

```python
# Emerging trends in robot language processing
EMERGING_TRENDS = {
    'large_language_model_integration': {
        'description': 'Using large pre-trained language models adapted for robotics',
        'impact': 'Improved understanding of complex and varied language',
        'research_directions': [
            'Fine-tuning LLMs for embodied tasks',
            'Prompt engineering for robot control',
            'Multimodal LLMs for vision-language-action integration'
        ],
        'timeline': 'Short to medium term'
    },
    'neuro_symbolic_approaches': {
        'description': 'Combining neural networks with symbolic reasoning for better systematicity',
        'impact': 'Improved generalization to novel combinations of known concepts',
        'research_directions': [
            'Differentiable symbolic reasoning',
            'Symbolic planning guided by neural networks',
            'Neural networks guided by symbolic knowledge'
        ],
        'timeline': 'Medium to long term'
    },
    'multimodal_foundation_models': {
        'description': 'Unified models that process language, vision, and action together',
        'impact': 'More coherent and effective integration of modalities',
        'research_directions': [
            'Joint training on language, vision, and action datasets',
            'Emergent capabilities through scaling',
            'Efficient adaptation to new environments'
        ],
        'timeline': 'Medium term'
    },
    'continual_learning': {
        'description': 'Systems that continuously learn from interaction without forgetting',
        'impact': 'Robots that improve over time and adapt to users',
        'research_directions': [
            'Catastrophic forgetting prevention',
            'Lifelong learning architectures',
            'Human-in-the-loop learning'
        ],
        'timeline': 'Long term'
    },
    'socially_aware_language': {
        'description': 'Language processing that considers social context and norms',
        'impact': 'More natural and appropriate human-robot interaction',
        'research_directions': [
            'Cultural adaptation of language understanding',
            'Social role recognition and appropriate responses',
            'Group interaction management'
        ],
        'timeline': 'Medium to long term'
    }
}

class FutureLanguageArchitecture:
    def __init__(self):
        self.large_language_model = LargeLanguageModelAdapter()
        self.multimodal_fusion = MultimodalFusionNetwork()
        self.neuro_symbolic_module = NeuroSymbolicModule()
        self.continual_learner = ContinualLearningModule()
        self.social_reasoner = SocialReasoningModule()
        
    def process_command_future(self, command, multimodal_context):
        """Process command using future architecture"""
        # Use large language model for semantic understanding
        semantic_representation = self.large_language_model.understand(command)
        
        # Fuse with visual and spatial context
        multimodal_representation = self.multimodal_fusion.integrate(
            semantic_representation,
            multimodal_context
        )
        
        # Apply neuro-symbolic reasoning for systematic processing
        formal_plan = self.neuro_symbolic_module.reason(
            multimodal_representation
        )
        
        # Consider social context
        socially_aware_plan = self.social_reasoner.adapt_to_social_context(
            formal_plan, multimodal_context.social_context
        )
        
        # Execute with continual learning
        execution_result = self.execute_with_learning(
            socially_aware_plan, multimodal_context
        )
        
        # Update continual learning system
        self.continual_learner.update_from_interaction(
            command, multimodal_context, execution_result
        )
        
        return execution_result
    
    def execute_with_learning(self, plan, context):
        """Execute plan while enabling continual learning"""
        # Execute the plan using robot capabilities
        result = self.execute_plan(plan, context)
        
        # Monitor execution for learning opportunities
        self.continual_learner.monitor_execution(
            plan, context, result
        )
        
        return result
    
    def execute_plan(self, plan, context):
        """Execute the plan in the real world"""
        # This would interface with robot execution system
        # Implementation would depend on specific robot platform
        return {'success': True, 'details': 'Plan executed successfully'}

# Research challenges for future robot language systems
RESEARCH_CHALLENGES = [
    {
        'challenge': 'Systematic Generalization',
        'description': 'Achieving systematic generalization to novel combinations of known concepts',
        'importance': 'Critical for open-ended interaction',
        'approach': 'Combine neural learning with symbolic reasoning'
    },
    {
        'challenge': 'Real-time Processing',
        'description': 'Processing complex language input in real-time with sufficient accuracy',
        'importance': 'Required for natural interaction',
        'approach': 'Efficient model architectures and hardware acceleration'
    },
    {
        'challenge': 'Ambiguity Resolution',
        'description': 'Resolving language ambiguity using contextual and world knowledge',
        'importance': 'Common in natural human language',
        'approach': 'Probabilistic reasoning and context modeling'
    },
    {
        'challenge': 'Safety and Reliability',
        'description': 'Ensuring safe behavior when language understanding is uncertain',
        'importance': 'Critical for human-robot interaction',
        'approach': 'Uncertainty quantification and safe exploration'
    },
    {
        'challenge': 'Multilingual and Cultural Adaptation',
        'description': 'Adapting to diverse languages and cultural contexts',
        'importance': 'Required for global deployment',
        'approach': 'Cultural knowledge integration and translation technologies'
    }
]
```

## Summary

Natural Language Processing for robot control enables robots to interact with humans using everyday language, making them more accessible and intuitive to use. This chapter explored the specialized challenges of processing natural language in the context of robot control, including spatial language processing, ambiguity resolution, and context awareness.

The key components of robot language systems include speech recognition, intent classification, spatial reference resolution, and action mapping. These systems must handle the unique aspects of robot-directed language, such as spatial references, deictic expressions, and the need to ground linguistic concepts in the physical environment.

Spatial language processing is particularly important for robots, as they must understand references to objects and locations in 3D space. This involves resolving references to specific environmental entities, handling different reference frames, and understanding spatial relationships.

Intent recognition requires hierarchical classification to handle the complex command structures common in robotics. Systems must also handle compound commands that require multiple coordinated actions, incorporating dependency and temporal relationship understanding.

The chapter also addressed multilingual support, which is important for robots deployed in diverse environments, and online learning, which enables systems to improve through continued interaction.

Evaluation of robot language systems requires metrics that consider both linguistic understanding and task execution success, as well as human factors like naturalness and social appropriateness.

As the field continues to advance, emerging trends in large language models, multimodal integration, and continual learning promise to create more capable and adaptive robot language systems.

## Exercises

1. Design a spatial reference resolution system that can ground expressions like "the cup on your left" or "the object behind the chair" in a real environment. What challenges would you face in implementing this system?

2. Implement an ambiguity resolution algorithm that can handle multiple types of linguistic ambiguity in robot commands. How would your system ask for clarification when needed?

3. Create a multilingual natural language interface for a robot that supports both English and Urdu. What specific challenges would arise when processing Urdu commands?

---

*This chapter is part of the Physical AI & Humanoid Robotics textbook. [Personalize Chapter] [Translate to Urdu]*