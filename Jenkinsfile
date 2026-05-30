// Full MLOps pipeline implementing the "Local MLOps Workflow Example" from MM2.
//
// Stages
//   1.  Checkout          — clone the repo
//   2.  Setup             — install uv + project dependencies on the agent
//   3.  Lint              — ruff check + format check
//   4.  Test              — pytest with coverage (fail fast gate)
//   5.  Validate          — assert required files / YAML integrity
//   6.  Docker Build      — build training image, tag with git commit hash
//   7.  Docker Push       — push both :<commit> and :latest to registry
//   8.  Train             — run train.py inside the container with lineage env vars
//   9.  Evaluate          — run evaluate.py; fails if accuracy < MIN_ACCURACY
//  10.  Register Model    — register model artifact to MLflow Model Registry (Staging)
//  11.  Deploy            — promote to Production and log deployment (main branch only)
//
// Branch protection:
//   Configure GitHub repo → Settings → Branches → Require status checks → "continuous-integration/jenkins/branch"
//   before merging into main.  The pipeline result is reported automatically by the
//   GitHub plugin (or Jenkins GitHub Checks plugin).
//
// Requirements
//   Jenkins agent with Python 3.12, Docker, and network access to REGISTRY and MLFLOW_URI.

pipeline {
    agent any

    environment {
        PYTHONUNBUFFERED   = "1"
        IMAGE_NAME         = "image-classifier"
        IMAGE_TAG          = "${env.GIT_COMMIT?.take(8) ?: 'latest'}"
        // Override DOCKER_REGISTRY / MLFLOW_URI in Jenkins → Manage Jenkins → Configure System
        // if your cluster addresses differ from the AAU defaults below.
        REGISTRY           = "${env.DOCKER_REGISTRY ?: '172.24.198.42:5000'}"
        MLFLOW_TRACKING_URI = "${env.MLFLOW_URI ?: 'http://172.24.198.42:5050'}"
        MIN_ACCURACY       = "0.80"
        // Set TRAIN_N_GPUS > 1 in Jenkins global config to enable multi-GPU DDP training.
        // Requires the Jenkins agent to have an NVIDIA runtime and nvidia-container-toolkit.
        TRAIN_N_GPUS       = "${env.TRAIN_N_GPUS ?: '1'}"
        PATH               = "${WORKSPACE}/.venv/bin:${env.PATH}"
    }

    stages {
        // ── 1. Checkout ─────────────────────────────────────────────────────────
        stage('Checkout') {
            steps {
                checkout scm
                echo "Branch: ${env.BRANCH_NAME} | Commit: ${env.GIT_COMMIT?.take(8)}"
            }
        }

        // ── 2. Setup ────────────────────────────────────────────────────────────
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

        // ── 3. Lint ─────────────────────────────────────────────────────────────
        stage('Lint') {
            steps {
                sh 'ruff check src/ tests/ train.py train_ddp.py evaluate.py inference.py scripts/'
                sh 'ruff format --check src/ tests/ train.py train_ddp.py evaluate.py inference.py scripts/'
            }
        }

        // ── 4. Test ─────────────────────────────────────────────────────────────
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

        // ── 5. Validate Structure ────────────────────────────────────────────────
        stage('Validate Structure') {
            steps {
                sh '''
                    set -e
                    for f in train.py evaluate.py inference.py pyproject.toml Dockerfile \
                              scripts/register_model.py scripts/log_deploy.py; do
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

        // ── 6. Docker Build ──────────────────────────────────────────────────────
        // Build a thin layer on top of the base image; tag with the git commit hash
        // so the exact environment is reproducible years from now.
        stage('Docker Build') {
            steps {
                sh "docker build -t ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} -t ${REGISTRY}/${IMAGE_NAME}:latest ."
                echo "Built ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
            }
        }

        // ── 7. Docker Push ───────────────────────────────────────────────────────
        // Push both the commit-hash tag (immutable lineage) and :latest (convenience).
        stage('Docker Push') {
            steps {
                sh "docker push ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
                sh "docker push ${REGISTRY}/${IMAGE_NAME}:latest"
                echo "Pushed ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
            }
        }

        // ── 8. Train ─────────────────────────────────────────────────────────────
        // Run training inside the exact image that was just pushed.
        // When TRAIN_N_GPUS > 1 the container launches train_ddp.py via torchrun
        // (DDP + AMP); otherwise falls back to the single-GPU train.py.
        // --gpus all exposes all host GPUs to the container (requires nvidia-container-toolkit).
        stage('Train') {
            steps {
                sh """
                    mkdir -p models/artifacts data/raw
                    GPU_FLAG=\$([ "${TRAIN_N_GPUS}" -gt 1 ] && echo "--gpus all" || echo "")
                    TRAIN_CMD=\$([ "${TRAIN_N_GPUS}" -gt 1 ] \
                        && echo "torchrun --standalone --nproc_per_node=${TRAIN_N_GPUS} train_ddp.py" \
                        || echo "python train.py")
                    docker run --rm \${GPU_FLAG} --shm-size=2g \\
                        -v \${WORKSPACE}/models:/app/models \\
                        -v \${WORKSPACE}/data:/app/data \\
                        -e MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \\
                        -e JENKINS_BUILD_NUMBER=${env.BUILD_NUMBER} \\
                        -e DOCKER_IMAGE_TAG=${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} \\
                        -e GIT_COMMIT_HASH=${env.GIT_COMMIT} \\
                        ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} \\
                        \${TRAIN_CMD}
                """
            }
        }

        // ── 9. Evaluate ──────────────────────────────────────────────────────────
        // Evaluate on the test split.  Exit code 1 (from --min-accuracy) fails the
        // stage immediately, preventing registration and deployment of a bad model.
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

        // ── 10. Register Model ───────────────────────────────────────────────────
        // Register the trained model artifact to the MLflow Model Registry and
        // move it to Staging.  Uses the run ID written by train.py.
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

        // ── 11. Deploy ───────────────────────────────────────────────────────────
        // Promote the Staging model to Production and log the deployment event back
        // to the training MLflow run.  Only runs on the main branch.
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
