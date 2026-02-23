# Update SubScript Version & Documentation

This skill prepares SubScript for a GitHub release by:
1. Updating the version number across configuration files
2. Re-reading the repository to sync skill documentation
3. Documenting any new functions in the function reference
4. Installing the package and running tests
5. Committing changes with appropriate message

## Workflow

When invoked, this skill will:

1. **Display current version** from `pyproject.toml`
2. **Prompt for new version number**
3. **Update version in:**
   - `./pyproject.toml`
   - `./meta.yaml`
   - `.claude/skills/galacticus-analysis/SKILL.md` (version badge in header)
4. **Detect new/changed modules** by checking `git status` for untracked or modified files in `subscript/`
5. **For each new/changed Python file**, read it and:
   - Add or update function entries in `.claude/skills/galacticus-analysis/references/subscript_functions.md`
   - Each entry should include: module path, function signature, description, parameters, returns, and a usage example
   - Also update the Summary Table at the bottom of that file
   - If the function fits an existing section in `SKILL.md`, add a usage snippet there too
6. **Update the `Last Updated` date** and version number at the top of `subscript_functions.md`
7. **If new features were detected** (new functions, new parameters, changed signatures), ask the user via `AskUserQuestion`:
   - "Yes" — write tests for the new features in the appropriate `tests/test_*.py` file, following the existing test style (mock data dicts, `numpy.testing.assert_equal`, etc.)
   - "No" — skip test writing
8. **Install package** in the `.venv` virtual environment:
   ```bash
   .venv/bin/python -m pip install -e .
   ```
9. **Run tests** to verify everything passes:
   ```bash
   .venv/bin/python -m pytest
   ```
   - If any tests fail, fix the issues before proceeding to commit
10. **Draft a commit message** and present it to the user using `AskUserQuestion`, offering options:
    - "Use as-is" — commit with the drafted message
    - "Edit message" — let the user provide a custom message
11. **Stage and commit** all changed files with the finalized message

## Files Updated

| File | What changes |
|------|-------------|
| `pyproject.toml` | `version = "X.Y.Z"` |
| `meta.yaml` | `version: "X.Y.Z"` |
| `.claude/skills/galacticus-analysis/SKILL.md` | Version badge, new function snippets |
| `.claude/skills/galacticus-analysis/references/subscript_functions.md` | Full function reference entries + Summary Table |

## Function Reference Format

When adding a new function to `subscript_functions.md`, follow this template:

```markdown
### function_name()

**Module:** `subscript.module_name`

**Signature:**
```python
def function_name(param1: type, param2: type = default, ...) -> return_type
```

**Description:**
One or two sentences describing what the function does.

**Parameters:**
- `param1` (type): Description
- `param2` (type, optional): Description. Default is X.

**Returns:**
- return_type: Description of returned value/structure

**Example:**
```python
from subscript.module_name import function_name

result = function_name(arg1, arg2)
```

**Notes:**
- Any important caveats or requirements
```

Add the new function to the **Summary Table** at the bottom:

```markdown
| `module_name` | `function_name()` | Category | One-line description |
```

## Automatic Git Commit

After updating all files, the skill will automatically:

1. **Display changed files** (using `git status`)
2. **Stage all updated files:**
   ```bash
   git add pyproject.toml meta.yaml \
     .claude/skills/galacticus-analysis/SKILL.md \
     .claude/skills/galacticus-analysis/references/subscript_functions.md \
     subscript/<new_module>.py   # any new source files
   ```
3. **Present the drafted commit message** to the user via `AskUserQuestion` with options to use it as-is or provide a custom message
4. **Create commit** with the finalized message:
   ```bash
   git commit -m "Bump version to X.Y.Z and add <module> module"
   ```

You can then push to GitHub:

```bash
git push
```

## Test Writing

When the user opts to write tests for new features:

- Add tests to the existing `tests/test_<module>.py` file that corresponds to the changed module (e.g., changes in `nfilters.py` → `tests/test_nfilters.py`)
- Follow the existing test style: mock data dicts, `numpy.testing.assert_equal`, simple assertions
- Test new parameters, new functions, and edge cases (e.g., shape mismatches raising `ValueError`)
- Include the new test file in the staged files for the commit

## Implementation Notes

- Current version is read from `pyproject.toml` (source of truth)
- Both `pyproject.toml` and `meta.yaml` must stay in sync
- Version format: `X.Y.Z` (semantic versioning)
- **Always** update `subscript_functions.md` when new `.py` files appear in `subscript/`
- The `Last Updated` date in `subscript_functions.md` should be set to today's date
- **Always** use `.venv/bin/python` for pip install and pytest — do not use system Python or conda
- All tests must pass before committing
- **Changes are automatically committed** - no manual git commands needed
- After commit, you only need to run `git push` to sync with GitHub
