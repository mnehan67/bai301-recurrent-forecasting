# Upload this project to your GitHub (Windows CMD)

These commands assume the downloaded ZIP was extracted in:

`C:\Users\muham\Downloads\xxxx`

After extraction, this folder should exist:

`C:\Users\muham\Downloads\xxxx\BAI301_RNN_GitHub_Actual_Run`

**1. Open Command Prompt (CMD)**. Press Windows, type `cmd`, press Enter.

**2. Go to the extracted repository folder.** Paste:

```bat
cd /d "C:\Users\muham\Downloads\xxxx\BAI301_RNN_GitHub_Actual_Run"
```

**3. Check Git and inspect the recorded outputs.** Paste the commands one at a time:

```bat
git --version
python scripts\audit_results.py
python scripts\check_run_provenance.py
```

If `git` is not recognized, install [Git for Windows](https://git-scm.com/download/win) and reopen CMD. The Python checks use the standard library and do not require GPUs.

**4. Create a NEW empty GitHub repository:**

- Sign in at [github.com/new](https://github.com/new).
- Name it `bai301-recurrent-forecasting` (or your preferred name).
- Choose **Public** or **Private** yourself. Start **Private** if author or licensing review is outstanding.
- **Do not select** “Add a README”, license, or `.gitignore`; those files already exist locally.
- Click **Create repository**.

**5. Initialize the local Git repo and review what will be uploaded:**

```bat
git init
git add .
git status
```

The file `local_only\Original_Kaggle_Results.zip` **must not** appear under files to commit. It is intentionally ignored because the archive includes redistributed electricity source data. The actual executed notebook and all run tables/logs **should** appear. You can verify exclusion with:

```bat
git check-ignore local_only\Original_Kaggle_Results.zip
```

**6. Commit and push:**

```bat
git commit -m "Add executed BAI301 notebook and full experiment results"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/bai301-recurrent-forecasting.git
git push -u origin main
```

**IMPORTANT:** Replace `YOUR_USERNAME` with your actual GitHub username. If you chose another repository name, change the end of the URL too. GitHub will usually open a sign-in window via Git Credential Manager. Avoid typing your password directly into CMD.

If Git asks you to identify yourself before the commit, run:

```bat
git config --global user.name "Muhammed Nehan"
git config --global user.email "YOUR_GITHUB_EMAIL"
```

Replace the example email, then rerun the `git commit` command.

**7. Verify:** Reload your GitHub repository webpage. Check that `README.md`, `notebooks/BAI301_Kaggle_T4x2_EXECUTED.ipynb`, `results/all_runs.csv`, and `paper/main.tex` are present. `local_only/` should not be there.

For later changes:

```bat
git add .
git commit -m "Update BAI301 experiment files"
git push
```

### If a push fails

- `remote origin already exists`: inspect `git remote -v`; update only if you are sure the URL is wrong.
- `src refspec main does not match any`: make sure `git commit` succeeded before pushing.
- `repository not found`: double-check your username and exact repo name in the remote URL.
- GitHub asks for authentication: use Git Credential Manager's browser flow or [GitHub's documented authentication methods](https://docs.github.com/en/authentication).
- You cannot see `local_only/` on GitHub: **expected**. Its raw archive stays on your computer.
