#!/bin/bash

# Update GitHub repository metadata for Quantum Trading AI

echo "📝 GitHub Repository Metadata Update Script"
echo "=========================================="
echo ""
echo "This script will help you update your GitHub repository with:"
echo "  - Description"
echo "  - Topics/Tags"
echo "  - Homepage URL"
echo "  - Branch protection rules"
echo ""

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed."
    echo ""
    echo "To install GitHub CLI:"
    echo "  - macOS: brew install gh"
    echo "  - Linux: See https://github.com/cli/cli#installation"
    echo ""
    echo "Alternatively, you can update these settings manually at:"
    echo "https://github.com/chandrashekargattu/quantum-trading-ai/settings"
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo "🔐 Please authenticate with GitHub CLI first:"
    echo "Run: gh auth login"
    exit 1
fi

echo "✅ GitHub CLI is installed and authenticated"
echo ""

# Repository settings
REPO_OWNER="chandrashekargattu"
REPO_NAME="quantum-trading-ai"

# Update repository description
echo "📋 Updating repository description..."
gh repo edit $REPO_OWNER/$REPO_NAME \
    --description "🚀 Advanced AI-powered options trading platform with quantum algorithms, ML models, real-time analysis, and automated trading strategies" \
    --homepage "https://quantum-trading-ai.com"

# Add topics/tags
echo "🏷️  Adding repository topics..."
gh repo edit $REPO_OWNER/$REPO_NAME \
    --add-topic "trading" \
    --add-topic "ai" \
    --add-topic "machine-learning" \
    --add-topic "quantum-computing" \
    --add-topic "options-trading" \
    --add-topic "fintech" \
    --add-topic "algorithmic-trading" \
    --add-topic "fastapi" \
    --add-topic "react" \
    --add-topic "nextjs" \
    --add-topic "python" \
    --add-topic "typescript" \
    --add-topic "real-time" \
    --add-topic "websocket" \
    --add-topic "backtesting"

# Create branch protection rules
echo "🔒 Setting up branch protection for 'main' branch..."
echo ""
echo "To set up branch protection rules, please visit:"
echo "https://github.com/$REPO_OWNER/$REPO_NAME/settings/branches"
echo ""
echo "Recommended settings for 'main' branch:"
echo "  ✓ Require pull request reviews before merging"
echo "  ✓ Dismiss stale pull request approvals when new commits are pushed"
echo "  ✓ Require status checks to pass before merging"
echo "  ✓ Require branches to be up to date before merging"
echo "  ✓ Include administrators"
echo "  ✓ Restrict who can push to matching branches"
echo ""

# Create useful GitHub files
echo "📁 Creating GitHub community files..."

# Create CONTRIBUTING.md
cat > CONTRIBUTING.md << 'EOF'
# Contributing to Quantum Trading AI

Thank you for your interest in contributing to Quantum Trading AI!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/quantum-trading-ai.git`
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `./run_all_tests.sh`
6. Commit with descriptive message: `git commit -m "feat: add new feature"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Code Style

- **Python**: Follow PEP 8, use Black formatter
- **TypeScript/React**: Use ESLint and Prettier
- **Commits**: Follow conventional commits format

## Testing

- Add tests for new features
- Ensure all tests pass before submitting PR
- Maintain or increase code coverage

## Pull Request Process

1. Update documentation
2. Add tests for new functionality
3. Ensure CI passes
4. Request review from maintainers
5. Address review feedback

## Code of Conduct

Please be respectful and constructive in all interactions.
EOF

# Create SECURITY.md
cat > SECURITY.md << 'EOF'
# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please:

1. **DO NOT** open a public issue
2. Email security concerns to: security@quantum-trading-ai.com
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will acknowledge receipt within 48 hours and provide updates on the resolution.

## Security Measures

- All dependencies are regularly updated
- Security scanning via GitHub Actions
- Input validation and sanitization
- Encrypted data transmission (TLS)
- Secure authentication (JWT)
- Rate limiting on APIs
EOF

# Create CODE_OF_CONDUCT.md
cat > CODE_OF_CONDUCT.md << 'EOF'
# Code of Conduct

## Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone.

## Our Standards

Examples of behavior that contributes to a positive environment:

* Using welcoming and inclusive language
* Being respectful of differing viewpoints
* Gracefully accepting constructive criticism
* Focusing on what is best for the community

Examples of unacceptable behavior:

* Harassment of any kind
* Trolling, insulting/derogatory comments
* Public or private harassment
* Publishing others' private information

## Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team. All complaints will be reviewed and investigated.

## Attribution

This Code of Conduct is adapted from the Contributor Covenant, version 2.0.
EOF

# Update README with badges
echo "🏅 Adding badges to README..."
echo ""
echo "Add these badges to your README.md:"
echo ""
echo '![Tests](https://github.com/chandrashekargattu/quantum-trading-ai/actions/workflows/ci-cd.yml/badge.svg)'
echo '![Code Quality](https://github.com/chandrashekargattu/quantum-trading-ai/actions/workflows/code-quality.yml/badge.svg)'
echo '![Security](https://github.com/chandrashekargattu/quantum-trading-ai/actions/workflows/dependency-update.yml/badge.svg)'
echo '[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)'
echo ""

# Create issue templates
mkdir -p .github/ISSUE_TEMPLATE

cat > .github/ISSUE_TEMPLATE/bug_report.md << 'EOF'
---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: 'bug'
assignees: ''
---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment:**
 - OS: [e.g. macOS]
 - Browser [e.g. chrome, safari]
 - Version [e.g. 22]

**Additional context**
Add any other context about the problem here.
EOF

cat > .github/ISSUE_TEMPLATE/feature_request.md << 'EOF'
---
name: Feature request
about: Suggest an idea for this project
title: '[FEATURE] '
labels: 'enhancement'
assignees: ''
---

**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is.

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.
EOF

# Create pull request template
cat > .github/pull_request_template.md << 'EOF'
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] E2E tests pass
- [ ] Manual testing completed

## Checklist
- [ ] My code follows the style guidelines
- [ ] I have performed a self-review
- [ ] I have commented my code where necessary
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix/feature works
- [ ] New and existing unit tests pass locally
EOF

echo ""
echo "✅ GitHub metadata update complete!"
echo ""
echo "📌 Next steps:"
echo "1. Visit https://github.com/$REPO_OWNER/$REPO_NAME/settings to verify settings"
echo "2. Set up branch protection rules (link above)"
echo "3. Add the badges to your README.md"
echo "4. Commit and push the new community files:"
echo "   git add ."
echo "   git commit -m 'docs: add GitHub community files and templates'"
echo "   git push"
