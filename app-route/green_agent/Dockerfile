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

# Make sudo dummy replacement that handles env vars properly.
RUN echo '#!/bin/bash\n\
if [ "$1" = "-E" ]; then shift; fi\n\
while [[ "$1" == *=* ]]; do export "$1"; shift; done\n\
"$@"' > /usr/bin/sudo && chmod +x /usr/bin/sudo


# Install Mininet and dependencies.
RUN apt-get update && apt-get install -y lsb-release git
RUN git clone https://github.com/mininet/mininet /opt/mininet
RUN cd /opt/mininet/util && DEBIAN_FRONTEND=noninteractive ./install.sh -a

# Install network utilities for Mininet (ping, traceroute, etc.)
RUN apt-get update && apt-get install -y iputils-ping net-tools iproute2 traceroute

WORKDIR /NetArena/app-route/green_agent

ENTRYPOINT ["./start_route_agent.sh"]
