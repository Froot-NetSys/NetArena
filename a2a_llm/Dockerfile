FROM ubuntu:22.04


WORKDIR /NetArena
COPY --chown=user . .

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

RUN uv sync --no-editable

WORKDIR /NetArena/a2a_llm

ENTRYPOINT ["uv", "run", "litellm_a2a_server.py"]