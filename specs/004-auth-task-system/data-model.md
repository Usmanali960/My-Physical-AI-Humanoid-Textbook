# Data Model: Better Auth Authentication & Reusable Intelligent Task System

## Entity: User
**Description**: Represents a person with authentication credentials, role assignments, and permissions for accessing AI features

**Fields**:
- `id` (UUID/Integer): Primary identifier
- `email` (String): User's email address, must be unique and valid
- `hashed_password` (String): Hashed password using secure algorithm (bcrypt)
- `first_name` (String, optional): User's first name
- `last_name` (String, optional): User's last name
- `oauth_provider` (String, optional): Provider name if registered via OAuth (Google, GitHub)
- `oauth_id` (String, optional): ID from OAuth provider
- `role` (String): User's role (admin, user)
- `is_active` (Boolean): Whether the account is active
- `created_at` (DateTime): When the account was created
- `updated_at` (DateTime): When the account was last updated
- `last_login` (DateTime, optional): When the user last logged in
- `permissions` (JSON/Text): List of permissions for the user (for fine-grained control)

**Relationships**:
- One-to-many with `Session` (a user can have multiple active sessions)
- One-to-many with `TaskExecutionLog` (a user can execute many tasks)

**Validation Rules**:
- Email must be a valid email format
- Password must meet complexity requirements (when applicable)
- Role must be one of the predefined roles (admin, user)
- Email must be unique across all users

## Entity: Role
**Description**: Defines a set of permissions that can be assigned to users, controlling their access to different features

**Fields**:
- `id` (UUID/Integer): Primary identifier
- `name` (String): Role name (admin, user, etc.)
- `description` (String): Description of the role
- `permissions` (JSON/Text): List of permissions associated with this role
- `created_at` (DateTime): When the role was created
- `updated_at` (DateTime): When the role was last updated

**Relationships**:
- Many-to-many with `User` (users can have multiple roles, roles can be assigned to multiple users)
- Many-to-many with `Permission` (roles can have multiple permissions, permissions can be part of multiple roles)

**Validation Rules**:
- Name must be unique
- Name must be from a predefined set of valid roles

## Entity: Permission
**Description**: Specifies what actions a user can perform within the system (e.g., access chatbot, run advanced tasks, administrative functions)

**Fields**:
- `id` (UUID/Integer): Primary identifier
- `name` (String): Permission name (e.g., "chatbot.access", "task.execute", "admin.manage")
- `description` (String): Description of what the permission allows
- `resource` (String): The resource the permission applies to
- `action` (String): The action that can be performed on the resource (read, write, delete, etc.)
- `created_at` (DateTime): When the permission was created
- `updated_at` (DateTime): When the permission was last updated

**Relationships**:
- Many-to-many with `Role` (roles can have multiple permissions, permissions can be part of multiple roles)

**Validation Rules**:
- Name must be unique
- Resource and action combination must be valid and predefined

## Entity: Session
**Description**: Contains authenticated user state and persists across requests until expiry or logout

**Fields**:
- `id` (UUID): Primary identifier (the session token)
- `user_id` (UUID/Integer): Reference to the user
- `expires_at` (DateTime): When the session expires
- `created_at` (DateTime): When the session was created
- `last_accessed` (DateTime): When the session was last used
- `user_agent` (String, optional): Browser/client information
- `ip_address` (String, optional): IP address from which session was created
- `is_active` (Boolean): Whether the session is still valid

**Relationships**:
- Many-to-one with `User` (multiple sessions per user)
- One-to-one with `SessionData` (optional additional session-specific data)

**Validation Rules**:
- Must reference a valid, active user
- Expiration time must be in the future
- Session token must be cryptographically secure and unique

## Entity: Task
**Description**: Represents a reusable AI-powered operation with standardized input/output schemas and execution logic

**Fields**:
- `id` (UUID/Integer): Primary identifier
- `name` (String): Name of the task
- `description` (String): Description of what the task does
- `input_schema` (JSON): Schema definition for required input parameters
- `output_schema` (JSON): Schema definition for expected output
- `auth_context_required` (Boolean): Whether authentication context injection is required
- `permissions_required` (JSON/Text): List of permissions required to execute this task
- `execution_logic` (String): Reference to the execution function/class
- `timeout` (Integer): Maximum execution time in seconds
- `created_at` (DateTime): When the task was created
- `updated_at` (DateTime): When the task was last updated
- `is_active` (Boolean): Whether the task is currently available for execution

**Relationships**:
- One-to-many with `TaskExecutionLog` (a task can be executed many times)

**Validation Rules**:
- Name must be unique
- Input and output schemas must be valid JSON Schema
- Execution logic must reference an existing, accessible function
- Timeout must be within reasonable limits

## Entity: TaskExecutionLog
**Description**: Records metadata about task executions for debugging and auditing purposes

**Fields**:
- `id` (UUID/Integer): Primary identifier
- `task_id` (UUID/Integer): Reference to the executed task
- `user_id` (UUID/Integer): Reference to the user who executed the task
- `auth_context` (JSON): Auth context at the time of execution
- `input_params` (JSON): Input parameters passed to the task
- `output_result` (JSON, optional): Output result of the task
- `execution_status` (String): Status of the execution (success, failed, timeout)
- `error_message` (String, optional): Error message if execution failed
- `execution_time_ms` (Integer): Time taken to execute in milliseconds
- `created_at` (DateTime): When the task execution started
- `completed_at` (DateTime, optional): When the task execution completed

**Relationships**:
- Many-to-one with `Task` (multiple executions per task)
- Many-to-one with `User` (multiple executions per user)

**Validation Rules**:
- Must reference a valid task and user
- Execution status must be one of the predefined values (success, failed, timeout)
- Execution time must be positive or null

## Entity: OAuthAccount
**Description**: Stores OAuth provider-specific information for users who register via OAuth

**Fields**:
- `id` (UUID/Integer): Primary identifier
- `user_id` (UUID/Integer): Reference to the user
- `provider` (String): OAuth provider name (Google, GitHub)
- `provider_account_id` (String): Account ID from the provider
- `provider_account_email` (String): Email from the provider
- `created_at` (DateTime): When the OAuth link was created
- `updated_at` (DateTime): When the link was last updated

**Relationships**:
- Many-to-one with `User` (multiple OAuth providers per user)

**Validation Rules**:
- User must be valid
- Provider must be one of the supported providers (Google, GitHub)
- Provider account ID must be unique per provider

## State Transitions

### User Account States
- `inactive` → `active` (after email verification)
- `active` → `suspended` (by admin action)
- `suspended` → `active` (by admin action)
- `active` → `deactivated` (by user request)

### Session States
- `active` → `expired` (when expiration time passes)
- `active` → `inactive` (when user logs out)
- `inactive` → `active` (when user logs back in)

### Task Execution States
- `pending` → `running` (when execution starts)
- `running` → `success` (when execution completes successfully)
- `running` → `failed` (when execution encounters an error)
- `running` → `timeout` (when execution exceeds timeout)
- `pending` → `cancelled` (when cancelled by user/admin)