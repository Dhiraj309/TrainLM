import sys

from aipm.cli.add import run_add
from aipm.cli.list import run_list


def main():
    if len(sys.argv) < 2:
        print("Usage: aipm <command> [args]")
        return

    command = sys.argv[1]

    if command == "add":
        if len(sys.argv) < 3:
            print("Usage: aipm add <path>")
            return

        run_add(sys.argv[2])
        return

    if command == "list":
        run_list()
        return

    print(f"Unknown command: {command}")
