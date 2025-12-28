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
    let vendor_dir = manifest_dir.parent().unwrap().join("vendor").join("cactus");
    let cactus_src_dir = vendor_dir.join("cactus"); // CMakeLists.txt is in vendor/cactus/cactus/

    // Check if Cactus source exists
    if !cactus_src_dir.exists() {
        panic!(
            "Cactus source not found at {:?}\n\
             Please run: git submodule update --init --recursive",
            cactus_src_dir
        );
    }

    // Build Cactus with CMake
    println!("cargo:rerun-if-changed=vendor/cactus");
    println!("cargo:rerun-if-changed=wrapper.h");

    // Configure and build (no install target available)
    let dst = cmake::Config::new(&cactus_src_dir)
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_BUILD_TYPE", "Release")
        .build_target("cactus") // Build only the static library target
        .build();

    // The library is built in out/build/ directory
    let lib_dir = dst.join("build");

    // Link directories
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // Also check for libcactus_pro.a
    let cactus_pro = vendor_dir.join("libs").join("libcactus_pro.a");
    if cactus_pro.exists() {
        println!(
            "cargo:rustc-link-search=native={}",
            vendor_dir.join("libs").display()
        );
        // Link cactus_pro first (whole archive for NPU symbols)
        #[cfg(target_os = "macos")]
        {
            println!("cargo:rustc-link-arg=-force_load");
            println!("cargo:rustc-link-arg={}", cactus_pro.display());
        }
    }

    // Link Cactus static library
    println!("cargo:rustc-link-lib=static=cactus");

    // Platform-specific frameworks/libraries
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=curl");
        println!("cargo:rustc-link-lib=c++");
    }

    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=curl");
    }

    // Generate bindings with bindgen
    let wrapper_path = manifest_dir.join("wrapper.h");

    let bindings = bindgen::Builder::default()
        .header(wrapper_path.to_str().unwrap())
        .clang_arg(format!("-I{}", cactus_src_dir.display()))
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++20")
        // Only generate bindings for cactus FFI functions
        .allowlist_function("cactus_.*")
        .allowlist_type("cactus_.*")
        .allowlist_var("CACTUS_.*")
        // Use core instead of std for no_std compatibility
        .use_core()
        // Derive common traits
        .derive_debug(true)
        .derive_default(true)
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
