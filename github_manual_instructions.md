# Manual GitHub Repository Setup Instructions

Since we're encountering HTTP 400 errors when trying to push automatically, please follow these manual steps to create your GitHub repository:

## Step 1: Create a new repository on GitHub.com

1. Go to https://github.com/new
2. Enter "lqr-inverted-pendulum" as the repository name
3. Add a description: "LQR Stabilization of an Inverted Pendulum: Advanced control systems & data analysis"
4. Choose Public visibility
5. DO NOT initialize with a README, .gitignore, or license
6. Click "Create repository"

## Step 2: Set up your local repository with SSH (recommended)

1. Check if you have an SSH key:
   ```
   cat ~/.ssh/id_rsa.pub
   ```

2. If you don't see a key, generate one:
   ```
   ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
   ```

3. Add your SSH key to GitHub:
   - Copy your public key: `cat ~/.ssh/id_rsa.pub`
   - Go to GitHub → Settings → SSH and GPG keys → New SSH key
   - Paste your key and save

4. Change your remote URL to use SSH:
   ```
   git remote set-url origin git@github.com:YOUR_USERNAME/lqr-inverted-pendulum.git
   ```
   (Replace YOUR_USERNAME with your GitHub username)

## Step 3: Push your code

1. First, make a single commit with just the essential files:
   ```
   git checkout -b clean-branch
   git add README.md pendulum_lqr.py requirements.txt .gitignore .gitattributes
   git commit -m "Initial commit with core files"
   git push -u origin clean-branch
   ```

2. Go to GitHub and verify that the branch was pushed successfully

3. Gradually add more files in separate commits:
   ```
   git add *.md *.py
   git commit -m "Add documentation and Python code"
   git push
   ```

4. If you want to include the binary files (images, CSV files, etc.):
   - Consider using Git LFS: https://git-lfs.github.com/
   - Or manually upload them through the GitHub web interface

## Alternative: Upload as a ZIP file

If you continue to experience issues with Git, you can:

1. Create a ZIP file of your project:
   ```
   zip -r lqr-inverted-pendulum.zip . -x "*.git*"
   ```

2. Upload the ZIP file to your GitHub repository using the web interface

## Final Note

The GitHub error (HTTP 400) usually occurs when pushing large files or many files at once. Using SSH instead of HTTPS and pushing in smaller increments can help avoid these issues. 