# MemFuse: The Lightning-Fast Memory Layer for LLMs ⚡️

![MemFuse Logo](https://img.shields.io/badge/MemFuse-OpenSource-blue.svg)  
[![Releases](https://img.shields.io/badge/Releases-Download%20Latest-brightgreen.svg)](https://github.com/sakhama/memfuse/releases)

Welcome to the **MemFuse** repository! This project provides the official core services for MemFuse, an open-source memory layer designed for large language models (LLMs). With MemFuse, you can give your AI applications a persistent and queryable memory, enhancing conversations across sessions.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [API Reference](#api-reference)
6. [Contributing](#contributing)
7. [License](#license)
8. [Support](#support)

## Introduction

In the age of artificial intelligence, memory plays a crucial role in how chatbots and conversational agents interact with users. MemFuse aims to bridge the gap by providing a seamless memory solution that enables LLMs to retain context and information over time. This enhances user experience and allows for more meaningful interactions.

## Features

- **Persistent Memory**: Store and retrieve information across multiple sessions.
- **Queryable Memory**: Efficiently query stored information to provide relevant responses.
- **Integration with LLMs**: Easily integrate with popular LLM frameworks.
- **Open Source**: Fully open-source, allowing for community contributions and transparency.
- **Lightweight**: Designed for speed and efficiency, minimizing resource usage.

## Installation

To get started with MemFuse, you need to install the Python SDK. Follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sakhama/memfuse.git
   cd memfuse
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.7 or higher installed. Then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and Execute the Latest Release**:
   Visit the [Releases](https://github.com/sakhama/memfuse/releases) section to download the latest version. Execute the downloaded file to set up MemFuse.

## Usage

Using MemFuse is straightforward. Here’s a simple example to get you started:

```python
from memfuse import MemFuse

# Initialize MemFuse
memory = MemFuse()

# Store a memory entry
memory.store("user_id_123", "User loves pizza.")

# Retrieve memory
response = memory.retrieve("user_id_123")
print(response)  # Output: User loves pizza.
```

This example demonstrates how to store and retrieve information using MemFuse. You can easily adapt this code to suit your application's needs.

## API Reference

### `MemFuse`

#### `__init__(self)`

Initializes a new instance of MemFuse.

#### `store(user_id: str, information: str)`

Stores information associated with a user ID.

- **Parameters**:
  - `user_id`: A unique identifier for the user.
  - `information`: The information to be stored.

#### `retrieve(user_id: str) -> str`

Retrieves stored information for a given user ID.

- **Parameters**:
  - `user_id`: A unique identifier for the user.
  
- **Returns**: The stored information as a string.

### Additional Methods

You can explore additional methods and functionalities in the [API documentation](#).

## Contributing

We welcome contributions to MemFuse! If you have ideas, bug fixes, or improvements, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature/YourFeature`).
6. Open a Pull Request.

Your contributions help improve MemFuse for everyone.

## License

MemFuse is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues or have questions, feel free to open an issue in the repository. For more immediate support, you can join our community on [Discord](https://discord.gg/memfuse).

For the latest releases, visit the [Releases](https://github.com/sakhama/memfuse/releases) section.

---

Thank you for checking out MemFuse! We look forward to seeing how you use it in your projects.