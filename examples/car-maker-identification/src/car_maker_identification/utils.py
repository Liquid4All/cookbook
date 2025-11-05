def print_installed_packages():
    # Print all installed packages with exact versions
    import pkg_resources

    print("\n=== All installed packages with versions ===")
    installed_packages = [d for d in pkg_resources.working_set]
    installed_packages.sort(key=lambda x: x.key.lower())
    for package in installed_packages:
        print(f"{package.key}=={package.version}")
    print("=" * 45)
