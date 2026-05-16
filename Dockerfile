# bedrockcache CLI — single-purpose container.
#
# Usage:
#   docker run --rm -i ghcr.io/mukundakatta/bedrockcache:latest audit - --backend litellm < request.json
#   echo '{...}' | docker run --rm -i ghcr.io/mukundakatta/bedrockcache:latest audit - --backend litellm --strict

FROM python:3.12-slim AS build

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN pip install --no-cache-dir build && python -m build --wheel --outdir /dist

FROM python:3.12-slim

RUN useradd --create-home --user-group --shell /usr/sbin/nologin bedrockcache
WORKDIR /home/bedrockcache

COPY --from=build /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/bedrockcache-*.whl && rm /tmp/*.whl

USER bedrockcache

ENTRYPOINT ["bedrockcache"]
CMD ["--help"]
