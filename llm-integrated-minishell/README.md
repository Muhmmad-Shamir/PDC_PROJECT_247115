Custom Linux Minishell with Local LLM Integration
A custom Linux shell written in C that implements core operating system principles and integrates a local Large Language Model (LLM) via Ollama.

🛡️ Cyber Security & Systems Context
Understanding shell architecture is fundamental to offensive security. This project demonstrates process isolation, secure subshell execution, and API manipulation. By utilizing a local TinyLlama instance instead of a public API, the tool maintains a hard data boundary, ensuring commands and system insights never leave the host environment.

🚀 Features
Process Management: Executes binaries via fork() and execvp().

IPC Pipes: Unidirectional kernel data streams via pipe() and dup2().

Backgrounding: Asynchronous tasks running via & with non-blocking WNOHANG reaping.

AI Engine (ask): Routes text-filtered queries to a local loopback Ollama daemon.

🏗️ Prerequisites & Setup
Before running the shell, ensure you have the necessary system utilities and the local inference engine configured:

Install system dependencies:

sudo apt update && sudo apt install curl jq gcc -y
Download and run the optimized local LLM:

ollama pull tinyllama
🚀 Compilation & Usage
Compile the source code:

gcc -o minishell minishell.c
Launch your custom shell:

./minishell
Query the local AI assistant:

minishell:~$ ask what is a buffer overflow attack
