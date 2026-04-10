// Jenkins declarative pipeline for the image-classifier fine-tuning project.
//
// Stages:
//   1. Checkout      — clone the repo
//   2. Setup         — install uv + project dependencies
//   3. Lint          — ruff check + format check
//   4. Test          — pytest with coverage
//   5. Validate      — assert required files / YAML integrity
//   6. Docker Build  — build the training image (main branch only)
//
// Requirements: Jenkins agent with Python 3.12 and Docker available.

pipeline {
    agent any

    environment {
        UV_SYSTEM_PYTHON = "1"          // uv will use the system Python
        PYTHONUNBUFFERED = "1"
        IMAGE_NAME = "image-classifier"
        IMAGE_TAG  = "${env.GIT_COMMIT?.take(8) ?: 'latest'}"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                echo "Branch: ${env.BRANCH_NAME} | Commit: ${env.GIT_COMMIT?.take(8)}"
            }
        }

        stage('Setup') {
            steps {
                sh '''
                    pip install uv --quiet --upgrade
                    uv pip install --system -e ".[dev]"
                '''
            }
        }

        stage('Lint') {
            steps {
                sh 'ruff check src/ tests/ train.py evaluate.py inference.py'
                sh 'ruff format --check src/ tests/ train.py evaluate.py inference.py'
            }
        }

        stage('Test') {
            steps {
                sh 'pytest tests/ -v --tb=short --junitxml=test-results.xml --cov=src --cov-report=xml'
            }
            post {
                always {
                    junit 'test-results.xml'
                    publishCoverage adapters: [coberturaAdapter('coverage.xml')]
                }
            }
        }

        stage('Validate Structure') {
            steps {
                sh '''
                    set -e
                    for f in train.py evaluate.py inference.py pyproject.toml Dockerfile; do
                        test -f "$f" || { echo "ERROR: missing $f"; exit 1; }
                    done
                    for d in src configs tests; do
                        test -d "$d" || { echo "ERROR: missing dir $d"; exit 1; }
                    done
                    python -c "
import yaml
for f in ['configs/base.yaml','configs/data.yaml','configs/model.yaml','configs/training.yaml']:
    yaml.safe_load(open(f))
    print(f'OK: {f}')
"
                    echo "Structure validated."
                '''
            }
        }

        stage('Docker Build') {
            when {
                branch 'main'
            }
            steps {
                sh "docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -t ${IMAGE_NAME}:latest ."
                echo "Built ${IMAGE_NAME}:${IMAGE_TAG}"
            }
        }
    }

    post {
        always {
            cleanWs()
        }
        success {
            echo "Pipeline succeeded."
        }
        failure {
            echo "Pipeline failed — check the logs above."
        }
    }
}
