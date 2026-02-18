# Update SubScript Version & Documentation

This skill prepares SubScript for a GitHub release by:
1. Updating the version number across configuration files
2. Re-reading the repository to sync skill documentation
3. Committing changes with appropriate message

## Workflow

When invoked, this skill will:

1. **Display current version** from `pyproject.toml`
2. **Prompt for new version number**
3. **Update version in:**
   - `./pyproject.toml`
   - `./meta.yaml`
4. **Re-sync documentation** by reading the repository and updating the galacticus-analysis skill
5. **Guide commit preparation**

## Usage

```bash
/update
```

The skill is interactive and will prompt you for the new version number.

## Automatic Git Commit

After updating all files, the skill will automatically:

1. **Display changed files** (using `git status`)
2. **Stage version files:**
   ```bash
   git add pyproject.toml meta.yaml .claude/skills/galacticus-analysis/SKILL.md
   ```
3. **Create commit** with semantic message:
   ```bash
   git commit -m "Bump version to X.Y.Z"
   ```

The commit is automatically staged and created. You can then push to GitHub:

```bash
git push
```

## Implementation Notes

- Current version is read from `pyproject.toml` (source of truth)
- Both `pyproject.toml` and `meta.yaml` must stay in sync
- Version format: `X.Y.Z` (semantic versioning)
- The skill auto-updates the galacticus-analysis skill documentation after version change
- **Changes are automatically committed** - no manual git commands needed
- After commit, you only need to run `git push` to sync with GitHub
