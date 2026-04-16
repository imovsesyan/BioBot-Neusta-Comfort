# GitHub Setup

This guide explains how to publish this clean repository to GitHub.

## Current Situation

This local folder is prepared as a clean repository:

```text
/Users/inesamovsesyan/Documents/Playground/BioBot-Neusta-Comfort
```

It is separate from the older `BioBot` folder so the new GitHub history can stay clean and focused.

## Recommended GitHub Repository Settings

Suggested repository name:

```text
biobot-neusta-comfort
```

Recommended visibility:

- Private, if the datasets or internship work are confidential.
- Public only if your manager confirms that the project can be shared.

Recommended description:

```text
Clean data preparation pipeline for BioBot / Neusta environmental comfort and livability prediction.
```

## Option A: Publish Manually with GitHub Website

1. Go to GitHub.
2. Click `New repository`.
3. Use the repository name:

```text
biobot-neusta-comfort
```

4. Do not add a README, `.gitignore`, or license on GitHub because this local repo already has them.
5. Create the repository.
6. Copy the repository URL.
7. In Terminal, run:

```bash
cd /Users/inesamovsesyan/Documents/Playground/BioBot-Neusta-Comfort
git init
git add .
git commit -m "Initial clean F8 data pipeline"
git branch -M main
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main
```

Replace `YOUR_GITHUB_REPO_URL` with the URL GitHub gives you.

## Option B: Let Codex Help Push

Codex can help after one of these is true:

1. You create an empty GitHub repo and give Codex the URL.
2. You install and authenticate GitHub CLI.

At the moment, `gh` is not installed in this environment. If you want CLI publishing, install it first:

```bash
brew install gh
gh auth login
```

Then Codex can help check the repo, commit, add the remote, and push.

## What Should Be Committed

Commit:

- source code in `src/`,
- runnable scripts in `scripts/`,
- English documentation in `docs/`,
- small summary JSON files in `reports/tables/`,
- small report figures in `reports/figures/`,
- tests,
- README and setup files.

Do not commit:

- raw data,
- generated CSV files,
- virtual environments,
- model binaries,
- cache files.

