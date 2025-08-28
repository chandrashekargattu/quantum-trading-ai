# 🧪 Testing Guide - Quantum Trading AI

This guide covers the testing setup and procedures for both backend and frontend of the Quantum Trading AI platform.

## 📋 Overview

- **Backend**: Python tests using pytest
- **Frontend**: JavaScript/TypeScript tests using Jest and React Testing Library
- **Coverage Requirements**: Minimum 80% for backend, 70% for frontend

## 🔧 Backend Testing

### Setup

1. **Install test dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt  # Already includes pytest and testing libs
   ```

2. **Set up test database**
   ```bash
   # Create test database
   createdb quantum_trading_test
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_auth.py

# Run specific test
pytest tests/test_auth.py::TestAuth::test_login_success

# Run with verbose output
pytest -v

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration
```

### Test Structure

```
backend/tests/
├── conftest.py          # Pytest configuration and fixtures
├── test_auth.py         # Authentication tests
├── test_users.py        # User management tests
├── test_portfolios.py   # Portfolio tests
├── test_trades.py       # Trading tests
├── test_market_data.py  # Market data tests
└── test_backtest.py     # Backtesting tests
```

### Key Test Fixtures

- `db_session`: Async database session for tests
- `client`: FastAPI test client
- `test_user`: Pre-created test user
- `test_portfolio`: Pre-created test portfolio
- `auth_headers`: Authentication headers with valid token

### Writing Backend Tests

```python
import pytest
from fastapi.testclient import TestClient

class TestExample:
    @pytest.mark.asyncio
    async def test_example(self, client: TestClient, auth_headers: dict):
        response = client.get("/api/v1/example", headers=auth_headers)
        assert response.status_code == 200
```

## 🎨 Frontend Testing

### Setup

1. **Install test dependencies**
   ```bash
   cd frontend
   npm install  # Already includes Jest and React Testing Library
   ```

### Running Tests

```bash
# Run all tests
npm test

# Run tests in watch mode
npm test -- --watch

# Run with coverage
npm test -- --coverage

# Run specific test file
npm test -- MarketOverview.test.tsx

# Update snapshots
npm test -- -u
```

### Test Structure

```
frontend/src/
├── __tests__/           # Global test files
├── components/
│   └── dashboard/
│       └── __tests__/   # Component tests
├── store/
│   └── __tests__/       # Store tests
├── services/
│   └── api/
│       └── __tests__/   # API service tests
├── lib/
│   └── __tests__/       # Utility tests
└── test-utils/
    └── test-utils.tsx   # Test utilities and helpers
```

### Key Test Utilities

- `render`: Custom render function with providers
- `mockUser`, `mockPortfolio`, etc.: Mock data generators
- `waitForLoadingToFinish`: Utility to wait for async operations

### Writing Frontend Tests

```typescript
import { render, screen, fireEvent } from '@/test-utils/test-utils'
import MyComponent from '../MyComponent'

describe('MyComponent', () => {
  it('renders correctly', () => {
    render(<MyComponent />)
    expect(screen.getByText('Expected Text')).toBeInTheDocument()
  })

  it('handles user interaction', async () => {
    render(<MyComponent />)
    
    const button = screen.getByRole('button', { name: 'Click me' })
    fireEvent.click(button)
    
    await waitFor(() => {
      expect(screen.getByText('Updated')).toBeInTheDocument()
    })
  })
})
```

## 🔄 Continuous Integration

### GitHub Actions Workflow

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  backend:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: quantum_trading_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        cd backend
        pip install -r requirements.txt
    - name: Run tests
      run: |
        cd backend
        pytest --cov=app --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  frontend:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Node
      uses: actions/setup-node@v3
      with:
        node-version: '20'
    - name: Install dependencies
      run: |
        cd frontend
        npm ci
    - name: Run tests
      run: |
        cd frontend
        npm test -- --coverage --ci
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## 📊 Coverage Reports

### Backend Coverage

After running tests with coverage:
```bash
# View in terminal
pytest --cov=app --cov-report=term-missing

# Generate HTML report
pytest --cov=app --cov-report=html
# Open htmlcov/index.html in browser
```

### Frontend Coverage

```bash
# Generate coverage report
npm test -- --coverage

# View coverage report
open coverage/lcov-report/index.html
```

## 🎯 Best Practices

1. **Test Isolation**: Each test should be independent
2. **Mock External Services**: Don't make real API calls in tests
3. **Use Fixtures**: Reuse common test data and setup
4. **Test Edge Cases**: Include error scenarios and boundary conditions
5. **Keep Tests Fast**: Mock heavy operations
6. **Descriptive Names**: Test names should clearly describe what they test
7. **AAA Pattern**: Arrange, Act, Assert

## 🐛 Debugging Tests

### Backend
```bash
# Run with print statements visible
pytest -s

# Run with debugger
pytest --pdb

# Run with specific log level
pytest --log-cli-level=DEBUG
```

### Frontend
```javascript
// Add debug statements
screen.debug()

// Use testing playground
screen.logTestingPlaygroundURL()

// Check what's rendered
console.log(screen.getByRole('button'))
```

## 📝 Test Checklist

Before pushing code:
- [ ] All tests pass locally
- [ ] New features have tests
- [ ] Test coverage meets requirements
- [ ] No console errors or warnings
- [ ] Tests are not flaky or dependent on timing
- [ ] Mocks are properly cleaned up

Happy Testing! 🚀
