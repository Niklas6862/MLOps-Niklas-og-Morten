pipeline {
    agent any

    parameters {
        booleanParam(
            name: 'USE_AMP',
            defaultValue: true,
            description: 'Use AMP (FP16) training script (train_amp.py). Uncheck to run the standard FP32 script (train.py).'
        )
        booleanParam(
            name: 'SKIP_COMPRESS',
            defaultValue: false,
            description: 'Skip the Compress stage (useful when testing AMP training only).'
        )
    }

    environment {
        PYTHONUNBUFFERED   = "1"
        IMAGE_NAME         = "image-classifier"
        IMAGE_TAG          = "${env.GIT_COMMIT?.take(8) ?: 'latest'}"
        REGISTRY           = "${env.DOCKER_REGISTRY ?: '172.24.198.42:5000'}"
        MLFLOW_TRACKING_URI = "${env.MLFLOW_URI ?: 'http://172.24.198.42:5050'}"
        MIN_ACCURACY       = "0.80"
        PATH               = "${WORKSPACE}/.venv/bin:${env.PATH}"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                echo "Branch: ${env.BRANCH_NAME} | Commit: ${env.GIT_COMMIT?.take(8)}"
                sh 'rm -rf models/'
            }
        }

        stage('Setup') {
            steps {
                sh '''
                    curl -LsSf https://astral.sh/uv/install.sh | sh
                    export PATH="$HOME/.local/bin:$PATH"
                    uv venv .venv
                    uv pip install -e ".[dev]"
                '''
            }
        }

        stage('Lint') {
            steps {
                sh 'ruff check src/ tests/ train.py train_amp.py evaluate.py inference.py compress.py batch_inference.py scripts/'
                sh 'ruff format --check src/ tests/ train.py train_amp.py evaluate.py inference.py compress.py batch_inference.py scripts/'
            }
        }

        stage('Test') {
            steps {
                sh 'pytest tests/ -v --tb=short --junitxml=test-results.xml --cov=src --cov-report=xml'
            }
            post {
                always {
                    junit 'test-results.xml'
                    archiveArtifacts artifacts: 'coverage.xml', allowEmptyArchive: true
                }
            }
        }

        stage('Validate Structure') {
            steps {
                sh '''
                    set -e
                    for f in train.py evaluate.py inference.py compress.py batch_inference.py \
                              pyproject.toml Dockerfile \
                              scripts/register_model.py scripts/log_deploy.py scripts/compress.sh; do
                        test -f "$f" || { echo "ERROR: missing $f"; exit 1; }
                    done
                    for d in src configs tests scripts; do
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
            steps {
                sh "docker build -t ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} -t ${REGISTRY}/${IMAGE_NAME}:latest ."
                echo "Built ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
            }
        }

        stage('Docker Push') {
            steps {
                sh "docker push ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
                sh "docker push ${REGISTRY}/${IMAGE_NAME}:latest"
                echo "Pushed ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
            }
        }

        stage('Train') {
            options { timeout(time: 60, unit: 'MINUTES') }
            steps {
                sh """
                    mkdir -p models/artifacts data/raw
                    TRAIN_SCRIPT=\$([ "${params.USE_AMP}" = "true" ] && echo "train_amp.py" || echo "train.py")
                    docker run --rm --gpus all --shm-size=2g --stop-timeout=5 \\
                        -v \${WORKSPACE}/models:/app/models \\
                        -v \${WORKSPACE}/data:/app/data \\
                        -e MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \\
                        -e JENKINS_BUILD_NUMBER=${env.BUILD_NUMBER} \\
                        -e DOCKER_IMAGE_TAG=${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} \\
                        -e GIT_COMMIT_HASH=${env.GIT_COMMIT} \\
                        ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} \\
                        python \${TRAIN_SCRIPT}
                """
            }
        }

        stage('Evaluate') {
            steps {
                sh """
                    docker run --rm \\
                        -v \${WORKSPACE}/models:/app/models \\
                        -v \${WORKSPACE}/data:/app/data \\
                        -e MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \\
                        ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} \\
                        python evaluate.py --min-accuracy ${MIN_ACCURACY}
                """
            }
        }

        stage('Compress') {
            when {
                not { expression { params.SKIP_COMPRESS } }
            }
            steps {
                sh """
                    docker run --rm \\
                        -v \${WORKSPACE}/models:/app/models \\
                        -v \${WORKSPACE}/data:/app/data \\
                        -e MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \\
                        ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} \\
                        python compress.py --method prune --prune-amount 0.3 --output models/artifacts/compression_report.json
                """
            }
            post {
                always {
                    archiveArtifacts artifacts: 'models/artifacts/compression_report.json', allowEmptyArchive: true
                }
            }
        }

        stage('Register Model') {
            steps {
                sh """
                    docker run --rm \\
                        -v \${WORKSPACE}/models:/app/models \\
                        -e MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \\
                        ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} \\
                        python scripts/register_model.py
                """
            }
        }

        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                sh """
                    docker run --rm \\
                        -v \${WORKSPACE}/models:/app/models \\
                        -e MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \\
                        -e JENKINS_BUILD_NUMBER=${env.BUILD_NUMBER} \\
                        ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} \\
                        python scripts/log_deploy.py
                """
                echo "Deployed ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} to Production."
            }
        }
    }

    post {
        always {
            cleanWs()
        }
        success {
            echo "Pipeline succeeded — model is live in Production (main) or Staging (dev)."
        }
        failure {
            echo "Pipeline FAILED — check the stage logs above and fix before re-merging."
        }
    }
}
