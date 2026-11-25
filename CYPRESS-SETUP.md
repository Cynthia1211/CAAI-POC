# CAAI-POC Cypress API Testing

## Overview

This project uses Cypress to test the Python FastAPI backend.

## Requirements

- Node.js 24 LTS
- npm
- Python backend dependencies
- Running FastAPI server at http://127.0.0.1:8000
- AWS access for live integration tests
- Database/Neptune connectivity for RAG tests

## Installation

Install JavaScript dependencies:

```bash
npm ci
```

## Running Tests

Run all tests:

```bash
npm test
```

Run only API contract tests:

```bash
npm run test:contract
```

Run live integration tests:

```bash
npm run test:live
```

## Test Suites

### backend-contract.cy.js

- TC-01: FastAPI documentation
- TC-02: Missing chatbot query
- TC-03: Empty speech synthesis text
- TC-04: Unsupported speech engine

### backend-live.cy.js

- TC-05: Retrieve Amazon Polly voices
- TC-06: Generate MP3 speech
- TC-07: Generate a RAG chatbot response

## Important Notes

- FastAPI must be running before executing tests.
- Live tests require configured AWS access.
- The RAG test requires approved synthetic test data.
- Live Polly requests may incur AWS usage charges.
- Never commit AWS credentials, .env files, or real patient data.

## Latest Local Results

Date: November 25, 2026

- Total tests: 7
- Passed: 7
- Failed: 0

These results were obtained locally and do not represent
a CI run or a guarantee of future test results.
