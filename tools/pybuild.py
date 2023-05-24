# Use this with CI to build for whatever version of python is currently configured in the
# runner.
import platform

def get_wheel_key():
    version = platform.python_version_tuple()
    key = "".join(version[0:-1])
    return f"cp{key}-manylinux_x86_64"

if __name__ == "__main__":
    # print(get_wheel_key(), end = "")
    print(get_wheel_key())
