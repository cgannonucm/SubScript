# Update SubScript Version & Documentation

This skill prepares SubScript for a GitHub release by:
1. Updating the version number across configuration files
2. Re-reading the repository to sync skill documentation
3. Documenting any new functions in the function reference
4. Committing changes with appropriate message

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
7. **Stage and commit** all changed files

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
3. **Create commit** with semantic message:
   ```bash
   git commit -m "Bump version to X.Y.Z and add <module> module"
   ```

You can then push to GitHub:

```bash
git push
```

## Implementation Notes

- Current version is read from `pyproject.toml` (source of truth)
- Both `pyproject.toml` and `meta.yaml` must stay in sync
- Version format: `X.Y.Z` (semantic versioning)
- **Always** update `subscript_functions.md` when new `.py` files appear in `subscript/`
- The `Last Updated` date in `subscript_functions.md` should be set to today's date
- **Changes are automatically committed** - no manual git commands needed
- After commit, you only need to run `git push` to sync with GitHub
