import subprocess
import pkg_resources
from packaging import version
import re

def get_installed_version(package_name):
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return None

def compare_versions(specified, installed):
    if specified.startswith('=='):
        return installed == specified[2:]
    elif specified.startswith('>='):
        return version.parse(installed) >= version.parse(specified[2:])
    # Add more comparisons here if needed
    return False

def parse_requirement(req):
    match = re.match(r'^([^<>=]+)(.*)$', req)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, None

def check_versions():
    with open('requirements.txt', 'r') as f:
        requirements = f.readlines()

    for req in requirements:
        req = req.strip()
        if req and not req.startswith('#'):
            package_name, specified_version = parse_requirement(req)
            if not package_name:
                print(f"Skipping invalid requirement: {req}")
                continue

            installed_version = get_installed_version(package_name)

            if installed_version:
                if compare_versions(specified_version, installed_version):
                    print(f"{package_name}: OK (installed: {installed_version}, requirement: {specified_version})")
                else:
                    print(f"{package_name}: MISMATCH (installed: {installed_version}, requirement: {specified_version})")
            else:
                print(f"{package_name}: NOT INSTALLED")

if __name__ == "__main__":
    check_versions()