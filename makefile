SONAR_URL   ?= http://sonarqube.sonarqube.svc.cluster.local:9000
SONAR_TOKEN ?= $(shell cat /config/.sonarqube-token 2>/dev/null)
SCANNER     ?= /config/.local/sonar-scanner/bin/sonar-scanner
JAVA_HOME   ?= $(shell dirname $$(dirname $$(readlink -f $$(which java))))
export JAVA_HOME

TAG     := $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
IMAGE   := contribute.void42.internal/golden/gmr-linguistics
PROJECT := gmr-linguistics
SRC     := src
TESTS   := tests

all: build release deploy

# ── Quality ──────────────────────────────────────────────────
test:
	python3 -m pytest tests/unit tests/component \
		--cov=$(SRC) \
		--cov-report=xml:coverage.xml \
		--cov-config=.coveragerc \
		--cov-fail-under=90 \
		--junitxml=test-results.xml \
		-q

test-integration:
	python3 -m pytest tests/integration -q

test-all: test test-integration

lint:
	python3 -m pylint $(SRC) $(TESTS) \
		--output-format=parseable --reports=no > pylint-report.txt 2>&1 || true
	@tail -1 pylint-report.txt

analyze: test lint
	$(SCANNER) \
		-Dsonar.projectKey=$(PROJECT) \
		-Dsonar.sources=$(SRC) \
		-Dsonar.tests=$(TESTS) \
		-Dsonar.language=py \
		-Dsonar.python.coverage.reportPaths=coverage.xml \
		-Dsonar.python.xunit.reportPath=test-results.xml \
		-Dsonar.python.pylint.reportPaths=pylint-report.txt \
		-Dsonar.host.url=$(SONAR_URL) \
		-Dsonar.token=$(SONAR_TOKEN) \
		-Dsonar.scm.provider=git \
		'-Dsonar.coverage.exclusions=src/backends/nllb_local.py,src/backends/labse_local.py,src/api/app.py,src/backends/base.py'
	@echo "Dashboard: $(SONAR_URL)/dashboard?id=$(PROJECT)"

security:
	pip-audit -r requirements.txt --desc 2>&1 || true
	pip-audit -r requirements-ml.txt --desc 2>&1 || true
	python3 -m bandit -r $(SRC) -q || true

# ── Deploy ───────────────────────────────────────────────────
build:
	docker build -t $(IMAGE):$(TAG) .

release:
	docker push $(IMAGE):$(TAG)

deploy:
	helm upgrade --install gmr-linguistics ./deployment --set-string version=$(TAG)
	kubectl -n gmr rollout restart deployment gmr-linguistics
	kubectl -n gmr rollout status deployment/gmr-linguistics --timeout=300s

# ── SBOM + Dependency-Track ─────────────────────────────────
DTRACK_URL  ?= http://dependency-track.dependency-track.svc.cluster.local:8080
DTRACK_KEY  ?= $(shell cat /config/.dtrack-api-key 2>/dev/null)

sbom:
	cyclonedx-py requirements requirements.txt --of json -o sbom.json
	curl -s -X POST "$(DTRACK_URL)/api/v1/bom" \
		-H "X-Api-Key: $(DTRACK_KEY)" \
		-H "Content-Type: multipart/form-data" \
		-F "autoCreate=true" \
		-F "projectName=$(PROJECT)" \
		-F "projectVersion=main" \
		-F "bom=@sbom.json" > /dev/null
	@echo "SBOM uploaded to Dependency-Track"

.PHONY: all test test-integration test-all lint analyze security build release deploy sbom
