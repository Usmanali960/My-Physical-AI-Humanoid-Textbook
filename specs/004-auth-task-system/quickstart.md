# Quickstart Guide: Better Auth Authentication & Reusable Intelligent Task System

## Overview

This guide will help you get started with the Better Auth Authentication & Reusable Intelligent Task System. This system provides secure authentication with role-based access control and a framework for executing reusable AI tasks.

## Prerequisites

- Python 3.11+
- pip package manager
- A Neon Postgres database (or compatible PostgreSQL database)
- API access to OAuth providers (Google, GitHub) if using OAuth

## Installation

1. Clone the repository or navigate to the backend directory:

```bash
cd backend-rag
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables by creating a `.env` file:

```bash
cp .env.example .env
```

Then update the `.env` file with your configuration:

```env
DATABASE_URL=postgresql://username:password@localhost:5432/your_database_name
NEON_DATABASE_URL=postgresql://username:password@ep-xxx.us-east-1.aws.neon.tech/your_database_name
SECRET_KEY=your_very_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GITHUB_CLIENT_ID=your_github_client_id
GITHUB_CLIENT_SECRET=your_github_client_secret
```

## Database Setup

1. Run database migrations:

```bash
alembic upgrade head
```

2. To create initial roles and permissions (optional):

```bash
python manage.py create_default_roles
```

## Running the Application

Start the development server:

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

## Making Your First API Call

### 1. Register a User

```bash
curl -X POST http://localhost:8000/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "securepassword123",
    "first_name": "John",
    "last_name": "Doe"
  }'
```

### 2. Login to Get a Session Token

```bash
curl -X POST http://localhost:8000/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "securepassword123"
  }'
```

You'll receive a response with a session token:

```json
{
  "success": true,
  "user": {
    "id": "user-uuid",
    "email": "user@example.com",
    "first_name": "John",
    "last_name": "Doe",
    "role": "user",
    "...": "..."
  },
  "session_token": "your-session-token-here",
  "message": "Login successful"
}
```

### 3. Use the Session Token for Authenticated Requests

Use the session token in the `Authorization` header:

```bash
curl -X GET http://localhost:8000/v1/auth/profile \
  -H "Authorization: Bearer your-session-token-here"
```

## Using the Task System

### 1. List Available Tasks

```bash
curl -X GET http://localhost:8000/v1/tasks \
  -H "Authorization: Bearer your-session-token-here"
```

### 2. Execute a Task

```bash
curl -X POST http://localhost:8000/v1/tasks/task-id-execute \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-session-token-here" \
  -d '{
    "param1": "value1",
    "param2": "value2"
  }'
```

## Admin Operations

Admin users can perform additional operations like managing users and roles. To get admin access, you may need to manually update the user's role in the database or use a special admin registration endpoint during initial setup.

### Update User Role (Admin Only)

```bash
curl -X PUT http://localhost:8000/v1/admin/users/user-id-in-here/role \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer admin-session-token-here" \
  -d '{
    "role": "admin"
  }'
```

## Adding New Tasks

To create a new reusable task:

1. Define the task schema in the `tasks/schemas.py` file
2. Create the task execution logic in a new file in the `tasks/` directory
3. Register the task in the task registry
4. Define the required permissions for the task

Example task definition:

```python
from pydantic import BaseModel
from tasks.base_task import BaseTask

# Define input and output schemas
class MyTaskInput(BaseModel):
    text: str
    count: int

class MyTaskOutput(BaseModel):
    result: str
    processed_items: int

# Create the task
class MyExampleTask(BaseTask[MyTaskInput, MyTaskOutput]):
    name = "my_example_task"
    description = "An example task that processes text"
    input_schema = MyTaskInput
    output_schema = MyTaskOutput
    permissions_required = ["task.execute"]
    
    async def execute(self, input_data: MyTaskInput, auth_context: dict) -> MyTaskOutput:
        # Your task logic here
        result = f"Processed: {input_data.text * input_data.count}"
        return MyTaskOutput(
            result=result,
            processed_items=input_data.count
        )
```

## Testing

Run the tests:

```bash
pytest
```

To run specific tests:

```bash
# Run only unit tests
pytest tests/unit/

# Run only contract tests
pytest tests/contract/

# Run with coverage
pytest --cov=backend-rag
```

## Configuration

The application uses a `config/settings.py` file to manage various settings. You can override these settings using environment variables.

Key configuration options:
- `DATABASE_URL`: Database connection string
- `SECRET_KEY`: Secret key for JWT tokens
- `ACCESS_TOKEN_EXPIRE_MINUTES`: Token expiration time
- `RATE_LIMIT_REQUESTS`: Number of requests allowed per window
- `RATE_LIMIT_WINDOW`: Time window for rate limiting (in seconds)

## Troubleshooting

### Common Issues:

1. **Database Connection Errors**: Verify your `DATABASE_URL` is correct and the database server is running.

2. **Authentication Fails**: Ensure you're sending the session token in the Authorization header as `Bearer {token}`.

3. **Task Execution Fails**: Check that you have the required permissions and the input parameters match the task's input schema.

4. **OAuth Issues**: Verify that your OAuth credentials are correct and that your redirect URLs are properly configured with the OAuth providers.

## Next Steps

1. Set up OAuth providers (Google, GitHub) if you plan to use them
2. Add custom tasks based on your specific AI features
3. Implement rate limiting for production use
4. Secure your API endpoints with HTTPS
5. Set up monitoring and logging for production