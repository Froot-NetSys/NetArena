FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Need Wget to install Miniconda.
RUN  apt-get update && apt-get install -y wget 

# Install Miniconda.
RUN mkdir -p /miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda3/miniconda.sh
RUN bash /miniconda3/miniconda.sh -b -u -p /miniconda3
RUN rm /miniconda3/miniconda.sh

ENV PATH="/miniconda3/bin:${PATH}"

WORKDIR /NetArena

# Build dependencies.
RUN apt-get install -y build-essential

# Create conda environments.
RUN conda tos accept
COPY --chown=user environment_ai_gym.yml .
RUN conda env create -f environment_ai_gym.yml
COPY --chown=user environment_mininet.yml .
RUN conda env create -f environment_mininet.yml

# Install Flash Attention separately since it requires PyTorch at build time...
RUN conda init --all
RUN conda run -n ai_gym_env pip install flash-attn==2.7.4.post1

# Make sudo dummy replacement, so we don't weaken docker security. Mininet (app-route) uses sudo as well.
RUN echo "#!/bin/bash\n\$@" > /usr/bin/sudo
RUN chmod +x /usr/bin/sudo

# Kubenertes CLI.
RUN apt-get install -y curl ca-certificates gnupg2
RUN curl -LO "https://dl.k8s.io/release/v1.34.0/bin/linux/amd64/kubectl"
RUN chmod +x ./kubectl
RUN mv ./kubectl /usr/local/bin/kubectl

# Add Docker's official GPG key:
RUN install -m 0755 -d /etc/apt/keyrings
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
RUN chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
RUN echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker CLI.
RUN apt-get update && apt-get install -y docker-ce-cli

# Install KIND.
RUN apt-get install -y golang-go
RUN go install sigs.k8s.io/kind@v0.26.0
RUN mv /root/go/ /usr/local/
ENV GOPATH=/usr/local/go
ENV PATH=$PATH:$GOPATH/bin

# Skaffold.
RUN curl -Lo skaffold https://storage.googleapis.com/skaffold/releases/v2.15.0/skaffold-linux-amd64
RUN chmod +x skaffold
RUN mv skaffold /usr/local/bin

# Make sudo dummy replacement, so we don't weaken docker security. Mininet (app-route) uses sudo as well.
RUN echo "#!/bin/bash\n\$@" > /usr/bin/sudo
RUN chmod +x /usr/bin/sudo

# Fetch Microservices repository and specify platform in loadgenerator Dockerfile.
RUN apt-get install -y git
RUN git clone https://github.com/GoogleCloudPlatform/microservices-demo.git
RUN sed -i 's|$BUILDPLATFORM|linux/amd64|g' microservices-demo/src/loadgenerator/Dockerfile

# Clean up apt cache to reduce image size.
RUN rm -rf /var/lib/apt/lists/*

COPY --chown=user . .

CMD ["/bin/bash"]