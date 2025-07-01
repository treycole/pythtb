# How to use Git

We will use the Gitflow workflow as described here

https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow

(Note: Atlassian promotes use of BitBucket, but we don't need it.)

Here are the steps that Sinisa did once and for all at the start:

1. Created a private GitHub repository.

2. Shared private GitHub repository with David through the GitHub website.

3. Created a develop branch and pushed it to GitHub:
```
  git branch develop
  git push -u origin develop
```

Note that "origin" in git language means GitHub repository.

4. Also made `master` and `develop` branches protected.

Below are instructions for creating new features, committing them, and merging changes into the `develop` branch.

## TL;DR Instructions

### Adding a feature
```bash
git checkout develop
git checkout -b feature_DESCRIPTOR
git add FILENAME
git commit -m "type message here"
git push -u origin feature_DESCRIPTOR
```

### Committing changes 

For every following commit, do

```bash
git add FILENAME
git commit -m "type message here"
git push origin
```

### Merging into `develop` 

On the Github website:
- Click 'New pull request'
- In pulldown menus: Set
  ```
      base:      develop  <<--- ESSENTIAL PART: use develop here!
      compare:   feature_DESCRIPTOR
  ```
- Write comments
- Click 'Create pull request'
- Maintainers merge into `develop` on the GitHub website

### Remove branch
```bash
git branch -d feature_DESCRIPTOR
git push origin --delete feature_DESCRIPTOR
```

## Step by step instructions
You want to add a feature to PythTB, here are the steps

1. Clone the repository:
  ```bash
    git clone http://github.com/sinisacoh/pythtb
  ```
  Or, if it is already in place, download updates
  ```bash
    git pull --all   # 'git pull' = 'git fetch' + 'git merge'
    # Option '--all' updates all branches
    git branch -a  # List of branches
    git status     # Info about current status
  ```
2. Switch to the develop branch:
  ```bash
    git checkout develop # or git switch develop in newer versions
  ```
3. Make a feature branch for the addition you want, starting from `develop`'s current state
  ```bash
    git checkout -b feature_DESCRIPTOR
  ```
  This will initially be a copy of `develop`.

4. Do whatever changes you want, then add the files to the git environment
  ```bash
    git add FILENAME
    ...
    git add -u   # Good for safety; adds all tracked files
    git status   # Check it looks OK
  ```
5. To submit changes to the feature branch remotely
  ```bash
    git commit -m "type message here"
    git push -u origin feature_DESCRIPTOR
  ```
  This last line has to be done once per local repository.  Next
  time you can simply do
  ```bash
    git push origin
  ```
  Note that any time you change any file, you need to `git add` it
  again before committing and pushing.

6. If someone else did some work on the same branch, and then did
  `add`/`commit`/`push_origin`, you can add those changes by doing
  ```bash
    git pull -u origin feature_DESCRIPTOR
  ```
7. When you are done working on the `feature_DESCRIPTOR` branch, and it has
  been pushed to origin, you should create a pull request to have
  it merged into the `develop` branch. On the GitHub website.
  - Click 'New pull request'
  - In pulldown menus: Set
  ```
      base:      develop  <<--- ESSENTIAL PART: use develop here!
      compare:   feature_DESCRIPTOR
  ```
  - Write comments
  - Click 'Create pull request'
  
See https://help.github.com/articles/creating-a-pull-request for more information.

As long as there are no conflicts, the maintainers merge the changes into
`develop` on the GitHub website. If there are conflicts, then they need to be resolved.

8. The `feature_DESCRIPTOR` branch can now be deleted:
  ```bash
    git branch -d feature_DESCRIPTOR                  ! locally
    git push origin --delete feature_DESCRIPTOR       ! remotely
  ```
9. When we have decided to issue a new release of the code, we should
    follow the steps described in [NEWVERSION](dev/NEWVERSION.md).
