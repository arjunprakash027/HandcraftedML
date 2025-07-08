# PPO Reinforcement Learning Environment

This project is set up to run a RL agents using OpenAI Gym and PyTorch. To accommodate different development environments, this repository provides separate Docker configurations for Linux (with NVIDIA GPU support) and macOS (CPU-based).

## Getting Started

Follow the instructions below to set up and run the development environment on your specific operating system.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Running the Environment

Choose the instructions that match your operating system.

---

#### For Linux (with NVIDIA GPU)

This setup enables GPU acceleration within the Docker container, which is highly recommended for training neural networks.

1.  **Build and Run the Container**

    Open a terminal in the project root and run the following command to build the Docker image and start the container in detached mode:

    ```bash
    docker-compose -f docker-compose.linux.yml up -d
    ```
---

#### For macOS (CPU / MPS)

This setup uses a standard CPU-based environment. It is compatible with both Intel and Apple Silicon (M1/M2) Macs.

1.  **Build and Run the Container**

    Open a terminal in the project root and run the following command:

    ```bash
    docker-compose -f docker-compose.macos.yml up -d
    ```
---

### Stopping the Environment

To stop the running containers, use the corresponding command for your OS:

-   **For Linux:**
    ```bash
    docker-compose -f docker-compose.linux.yml down
    ```

-   **For macOS:**
    ```bash
    docker-compose -f docker-compose.macos.yml down
    ```
