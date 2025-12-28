//! Build script for cactus-sys
//!
//! This script:
//! 1. Builds Cactus C++ library using CMake
//! 2. Generates Rust FFI bindings using bindgen
//! 3. Links the static library

use std::env;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cactus_dir = manifest_dir.parent().unwrap().join("vendor").join("cactus");

    // Check if Cactus source exists
    if !cactus_dir.exists() {
        panic!(
            "Cactus source not found at {:?}\n\
             Please run: git submodule update --init --recursive",
            cactus_dir
        );
    }

    // Build Cactus with CMake
    println!("cargo:rerun-if-changed=vendor/cactus");
    println!("cargo:rerun-if-changed=wrapper.h");

    let mut cmake_config = cmake::Config::new(&cactus_dir);

    cmake_config
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CACTUS_BUILD_TESTS", "OFF")
        .define("CACTUS_BUILD_EXAMPLES", "OFF");

    // Platform-specific settings
    #[cfg(target_os = "macos")]
    {
        cmake_config.define("CACTUS_METAL", "ON");
        cmake_config.define("CACTUS_ACCELERATE", "ON");
    }

    #[cfg(target_os = "linux")]
    {
        cmake_config.define("CACTUS_METAL", "OFF");
        cmake_config.define("CACTUS_OPENBLAS", "ON");
    }

    let dst = cmake_config.build();

    // Link directories
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());

    // Link Cactus static library
    println!("cargo:rustc-link-lib=static=cactus");

    // Platform-specific frameworks/libraries
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=c++");
    }

    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=pthread");
    }

    // Generate bindings with bindgen
    let wrapper_path = manifest_dir.join("wrapper.h");

    let bindings = bindgen::Builder::default()
        .header(wrapper_path.to_str().unwrap())
        .clang_arg(format!("-I{}", cactus_dir.join("cactus").display()))
        .clang_arg(format!("-I{}/include", dst.display()))
        // Only generate bindings for cactus FFI functions
        .allowlist_function("cactus_.*")
        .allowlist_type("cactus_.*")
        .allowlist_var("CACTUS_.*")
        // Use core instead of std for no_std compatibility
        .use_core()
        // Generate rustified enums
        .rustified_enum(".*")
        // Derive common traits
        .derive_debug(true)
        .derive_default(true)
        .derive_eq(true)
        .derive_hash(true)
        // Generate documentation
        .generate_comments(true)
        .generate()
        .expect("Unable to generate bindings");

    // Write bindings to OUT_DIR
    let bindings_path = out_dir.join("bindings.rs");
    bindings
        .write_to_file(&bindings_path)
        .expect("Couldn't write bindings!");

    println!("cargo:bindings_path={}", bindings_path.display());
}
