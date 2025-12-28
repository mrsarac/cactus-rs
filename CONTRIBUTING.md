# Contributing to cactus-rs

Thank you for your interest in contributing to cactus-rs! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Developer Certificate of Origin (DCO)](#developer-certificate-of-origin-dco)
- [Coding Standards](#coding-standards)
- [Testing](#testing)

## Code of Conduct

This project follows the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct). Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone --recursive https://github.com/YOUR_USERNAME/cactus-rs
   cd cactus-rs
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/mrsarac/cactus-rs
   ```

## Development Setup

### Prerequisites

- Rust 1.70+ (stable)
- CMake 3.14+
- Clang (for bindgen)

### macOS

```bash
xcode-select --install
brew install cmake llvm
```

### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y cmake clang libclang-dev build-essential
```

### Building

```bash
# Build all crates
cargo build

# Build with release optimizations
cargo build --release

# Build documentation
cargo doc --open
```

### Running Tests

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_message_system
```

## Making Changes

1. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make your changes** following the [coding standards](#coding-standards)

3. **Test your changes**:
   ```bash
   cargo test
   cargo clippy -- -D warnings
   cargo fmt --check
   ```

4. **Commit with DCO sign-off** (see [DCO section](#developer-certificate-of-origin-dco))

## Pull Request Process

1. **Update documentation** if you're changing public APIs
2. **Add tests** for new functionality
3. **Ensure CI passes** - all checks must be green
4. **Write a clear PR description** explaining:
   - What changes you made
   - Why you made them
   - How to test them
5. **Request review** from maintainers

### PR Title Format

Use conventional commit style:
- `feat: add new embedding method`
- `fix: handle null pointer in transcribe`
- `docs: update README examples`
- `test: add VectorIndex tests`
- `refactor: simplify error handling`

## Developer Certificate of Origin (DCO)

We use the Developer Certificate of Origin (DCO) as a way to manage contributions. The DCO is a legally binding statement that asserts you have the right to submit your contribution.

### Signing Your Commits

All commits must be signed off. Add `-s` or `--signoff` to your commit:

```bash
git commit -s -m "feat: add awesome feature"
```

This adds a `Signed-off-by` line to your commit message:

```
feat: add awesome feature

Signed-off-by: Your Name <your.email@example.com>
```

### Configuring Git for Sign-off

Set up your Git identity:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### DCO Text

By signing off, you agree to the following:

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

### Fixing Missing Sign-offs

If you forgot to sign off your commits:

```bash
# Last commit
git commit --amend --signoff

# Multiple commits
git rebase --signoff HEAD~N  # where N is the number of commits
```

## Coding Standards

### Rust Style

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `cargo fmt` for formatting
- Use `cargo clippy` for linting
- No warnings allowed in CI

### Documentation

- Document all public items with `///` doc comments
- Include examples in doc comments where helpful
- Keep comments concise and up-to-date

### Error Handling

- Use the crate's `Error` type for errors
- Provide meaningful error messages
- Use `Result<T>` for fallible operations

### Safety

- Minimize `unsafe` code
- Document all safety invariants
- Prefer safe abstractions

## Testing

### Unit Tests

Add tests in the same file using `#[cfg(test)]`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature() {
        // Your test here
    }
}
```

### Integration Tests

Add integration tests in `tests/` directory.

### Test Coverage

We aim for high test coverage. Run tests with coverage:

```bash
cargo install cargo-tarpaulin
cargo tarpaulin --out Html
```

## Questions?

- Open an issue for questions
- Join discussions in pull requests
- Contact maintainers: [@mrsarac](https://github.com/mrsarac)

---

Thank you for contributing to cactus-rs!
