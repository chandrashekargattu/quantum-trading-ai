# Quantum Trading AI - Test Suite Documentation

## Overview

This document provides comprehensive information about the test suite for the Quantum Trading AI platform, covering both frontend and backend testing strategies.

## Test Structure

### Frontend Tests

Located in: `frontend/src/app/**/__tests__/`

#### Authentication Tests

1. **Login Page Tests** (`frontend/src/app/auth/login/__tests__/login.test.tsx`)
   - ✅ Renders login form correctly
   - ✅ Displays branding panel
   - ✅ Validates empty fields
   - ✅ Toggles password visibility
   - ✅ Handles successful login
   - ✅ Handles login failures
   - ✅ Handles network errors
   - ✅ Disables form during submission
   - ✅ Links to registration and forgot password

2. **Registration Page Tests** (`frontend/src/app/auth/register/__tests__/register.test.tsx`)
   - ✅ Renders registration form correctly
   - ✅ Displays branding panel with features
   - ✅ Validates password match
   - ✅ Validates password length
   - ✅ Toggles password visibility
   - ✅ Handles successful registration
   - ✅ Handles registration failures
   - ✅ Validates email format
   - ✅ Links to terms and privacy policy

### Backend Tests

Located in: `backend/tests/`

#### Authentication Tests (`backend/tests/test_auth.py`)

1. **Registration Endpoints**
   - ✅ Register new user
   - ✅ Prevent duplicate email registration
   - ✅ Validate email format
   - ✅ Enforce password strength

2. **Login Endpoints**
   - ✅ Login with valid credentials
   - ✅ Reject invalid email
   - ✅ Reject invalid password
   - ✅ Block inactive users

3. **User Management**
   - ✅ Get current user info
   - ✅ Require authentication
   - ✅ Handle expired tokens
   - ✅ Refresh access tokens
   - ✅ Change password
   - ✅ Password reset flow

4. **Security Tests**
   - ✅ Password hashing security
   - ✅ Token validation
   - ✅ Authorization headers

## Running Tests

### Quick Start

Run all tests with a single command:

```bash
./run_all_tests.sh
```

### Frontend Tests Only

```bash
cd frontend
npm test
```

To run specific test files:
```bash
npm test -- login.test.tsx
npm test -- register.test.tsx
```

To run with coverage:
```bash
npm test -- --coverage
```

### Backend Tests Only

```bash
cd backend
python -m pytest
```

To run specific test files:
```bash
python -m pytest tests/test_auth.py -v
python -m pytest tests/test_market_data.py -v
```

To run with coverage:
```bash
python -m pytest --cov=app --cov-report=html
```

## Test Configuration

### Frontend Configuration

- **Test Framework**: Jest + React Testing Library
- **Configuration File**: `frontend/jest.config.js`
- **Test Environment**: jsdom
- **Coverage Threshold**: 80%

### Backend Configuration

- **Test Framework**: pytest + pytest-asyncio
- **Configuration File**: `backend/pytest.ini`
- **Test Database**: SQLite (in-memory)
- **Fixtures**: `backend/tests/conftest.py`

## Test Fixtures

### Frontend Mocks
- `next/navigation` - Mocked router
- `fetch` - Mocked API calls
- `localStorage` - Cleared between tests

### Backend Fixtures
- `db` - Test database session
- `client` - Test HTTP client
- `test_user` - Regular user fixture
- `test_superuser` - Admin user fixture
- `auth_headers` - Authentication headers
- `test_portfolio` - Sample portfolio
- `test_stock` - Sample stock data
- `test_option` - Sample option data

## Coverage Reports

### Frontend Coverage
After running tests, view coverage at:
- Terminal: Displayed after test run
- HTML Report: `frontend/coverage/lcov-report/index.html`

### Backend Coverage
After running tests, view coverage at:
- Terminal: Displayed after test run
- HTML Report: `backend/htmlcov/index.html`

## Best Practices

### Writing Frontend Tests

1. **Use Testing Library queries**
   ```typescript
   // Good
   screen.getByRole('button', { name: /sign in/i })
   
   // Avoid
   screen.getByTestId('signin-button')
   ```

2. **Test user behavior, not implementation**
   ```typescript
   // Good
   await user.type(emailInput, 'test@example.com')
   await user.click(submitButton)
   
   // Avoid
   fireEvent.change(emailInput, { target: { value: 'test@example.com' } })
   ```

3. **Use async/await for asynchronous operations**
   ```typescript
   await waitFor(() => {
     expect(mockPush).toHaveBeenCalledWith('/dashboard')
   })
   ```

### Writing Backend Tests

1. **Use async fixtures**
   ```python
   @pytest.mark.asyncio
   async def test_endpoint(client: AsyncClient):
       response = await client.get("/api/v1/endpoint")
   ```

2. **Test edge cases**
   ```python
   # Test with invalid data
   # Test with missing authentication
   # Test with expired tokens
   ```

3. **Clean up test data**
   ```python
   # Use transactions that rollback
   # Clean up files after tests
   ```

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - run: cd frontend && npm install
      - run: cd frontend && npm test

  backend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: cd backend && pip install -r requirements.txt
      - run: cd backend && pip install -r requirements-test.txt
      - run: cd backend && python -m pytest
```

## Troubleshooting

### Common Frontend Test Issues

1. **Module not found errors**
   - Ensure all imports have proper mocks
   - Check `jest.config.js` module mappings

2. **Act warnings**
   - Wrap state updates in `act()`
   - Use `waitFor` for async operations

### Common Backend Test Issues

1. **Database connection errors**
   - Ensure test database is properly configured
   - Check async context managers

2. **Import errors**
   - Verify PYTHONPATH includes project root
   - Check relative imports

## Future Improvements

1. **E2E Testing**
   - Add Cypress or Playwright for end-to-end tests
   - Test complete user workflows

2. **Performance Testing**
   - Add load testing with Locust
   - Monitor response times

3. **Security Testing**
   - Add OWASP ZAP integration
   - Test for common vulnerabilities

4. **Visual Regression Testing**
   - Add Percy or Chromatic
   - Catch UI regressions

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain > 80% code coverage
4. Update this documentation
