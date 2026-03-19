FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY requirements-lock.txt /workspace/requirements-lock.txt
RUN pip install --no-cache-dir -r /workspace/requirements-lock.txt

COPY . /workspace
RUN pip install --no-cache-dir -e .

CMD ["python", "-m", "scripts.reproduce", "--output-root", "artifacts/container_run"]
