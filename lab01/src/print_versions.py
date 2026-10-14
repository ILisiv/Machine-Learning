import sys
import pkg_resources

print(f"Python version: {sys.version}\n")
print("Installed Packages:")
for dist in sorted(pkg_resources.working_set, key=lambda x: x.key):
    print(f"{dist.key}=={dist.version}")