FROM ubuntu:22.04


WORKDIR /NetPress
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

# Kubenertes CLI.
RUN apt-get install -y curl ca-certificates gnupg2
RUN curl -LO "https://dl.k8s.io/release/v1.34.0/bin/linux/amd64/kubectl"
RUN chmod +x ./kubectl
RUN mv ./kubectl /usr/local/bin/kubectl

WORKDIR /NetPress/app-k8s/green_agent

ENTRYPOINT ["uv", "run", "k8s_agent.py"]
