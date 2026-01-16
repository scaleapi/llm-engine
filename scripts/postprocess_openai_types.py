#!/usr/bin/env python3
"""
Post-process generated OpenAI types to fix Pydantic compatibility issues.

This script fixes issues that arise from datamodel-codegen's strict OpenAPI spec
interpretation that don't work well with Pydantic's discriminated unions.
"""

import re
import sys
from pathlib import Path

# Unions where discriminator='type' must be removed because multiple variants
# share the same discriminator value (e.g., InputMessage and OutputMessage both
# have type='message').
#
# Without removal, Pydantic raises:
#   TypeError: Value 'message' for discriminator 'type' mapped to multiple choices
#
# These unions will fall back to trying each model in order until one validates.
UNIONS_WITH_CONFLICTING_DISCRIMINATORS = {
    "Item",       # InputMessage (type='message') + OutputMessage (type='message')
    "InputItem",  # EasyInputMessage + InputMessage + OutputMessage (all type='message')
}


def remove_discriminator_from_field(content: str, class_name: str) -> str:
    """
    Remove discriminator='type' from a specific RootModel class's Field() annotation.

    Handles these patterns:
    - Field(discriminator='type')  -> Field()
    - Field(..., discriminator='type')  -> Field(...)
    - Field(discriminator='type', ...)  -> Field(...)
    - Multi-line Field() with discriminator='type' on separate line
    """
    lines = content.split('\n')
    result_lines = []
    in_target_class = False

    for line in lines:
        # Check if we're leaving the class (new class at same or lower indent)
        # Must check BEFORE entry to avoid exiting on the same line we enter
        if in_target_class:
            if line.startswith('class ') and not line.startswith(' '):
                in_target_class = False

        # Check if we're entering a target class (exact match to avoid InputItem matching Item)
        # Matches: "class Item(" or "class Item(RootModel..."
        if re.match(rf'^class {class_name}\($', line) or re.match(rf'^class {class_name}\(RootModel', line):
            in_target_class = True

        # If in target class, remove discriminator from Field()
        if in_target_class:
            # Handle single-line patterns
            line = re.sub(r"Field\(discriminator='type'\)", "Field()", line)
            # Handle discriminator on its own line or with trailing comma
            line = re.sub(r"^\s*discriminator='type',?\s*$", "", line)
            # Handle inline discriminator with other args
            line = re.sub(r"discriminator='type',\s*", "", line)
            line = re.sub(r",\s*discriminator='type'", "", line)

        result_lines.append(line)

    return '\n'.join(result_lines)


def postprocess_server_types(filepath: Path) -> None:
    """Post-process server-side generated types (pydantic v2)."""
    content = filepath.read_text()
    original = content

    # 1. Replace pydantic import with custom module (for AnyUrl types)
    content = re.sub(
        r'^from pydantic import ',
        'from model_engine_server.common.pydantic_types import ',
        content,
        flags=re.MULTILINE
    )

    # 2. Remove discriminator from unions with conflicting type values
    for class_name in UNIONS_WITH_CONFLICTING_DISCRIMINATORS:
        content = remove_discriminator_from_field(content, class_name)

    if content != original:
        filepath.write_text(content)
        print(f"Post-processed {filepath}")


def postprocess_client_types(filepath: Path) -> None:
    """Post-process client-side generated types (pydantic v1/v2 compatible)."""
    content = filepath.read_text()
    original = content

    # 1. Add mypy ignore at the top
    if not content.startswith("# mypy: ignore-errors"):
        content = "# mypy: ignore-errors\n" + content

    # 2. Add conditional import for pydantic v1 and v2
    pydantic_import_pattern = r'^from pydantic import (.*)$'
    match = re.search(pydantic_import_pattern, content, re.MULTILINE)
    if match:
        imports = match.group(1)
        replacement = f'''import pydantic
PYDANTIC_V2 = hasattr(pydantic, "VERSION") and pydantic.VERSION.startswith("2.")
if PYDANTIC_V2:
    from pydantic.v1 import {imports}  # noqa: F401
else:
    from pydantic import {imports}  # type: ignore # noqa: F401'''
        content = re.sub(pydantic_import_pattern, replacement, content, flags=re.MULTILINE)

    # 3. Remove discriminator from unions with conflicting type values
    for class_name in UNIONS_WITH_CONFLICTING_DISCRIMINATORS:
        content = remove_discriminator_from_field(content, class_name)

    if content != original:
        filepath.write_text(content)
        print(f"Post-processed {filepath}")


def main():
    if len(sys.argv) < 3:
        print("Usage: postprocess_openai_types.py <server|client> <filepath>")
        sys.exit(1)

    mode = sys.argv[1]
    filepath = Path(sys.argv[2])

    if not filepath.exists():
        print(f"Error: {filepath} does not exist")
        sys.exit(1)

    if mode == "server":
        postprocess_server_types(filepath)
    elif mode == "client":
        postprocess_client_types(filepath)
    else:
        print(f"Error: unknown mode '{mode}', expected 'server' or 'client'")
        sys.exit(1)


if __name__ == "__main__":
    main()
