We will use  gitflow workflow as described here

https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow

(Note: Atlassian promotes use of BitBucket, but we don't need it.)

*************************

Here are steps that Sinisa did once and for all at the start:

1. Created private github repository.

2. Shared private github repository with David through github website.

3. Created develop branch and pushed it to github:
```
  git branch develop
  git push -u origin develop
```

Note that "origin" in git language means github repository.

4. Also made master and develop branch protected.

*************************

1. If David wants to add a feature to pythtb he should first get
the repository:
```
  git clone http://github.com/sinisacoh/pythtb
```
or, if it already in place, download updates
```
  git pull --all          'git pull' = 'git fetch' + 'git merge'
                          Option '--all' updates all branches
  git branch -a           List of branches
  git status              Info about current status
```
2. Then David should change to the develop branch:
```
  git checkout develop
```
3. Next David should make a feature branch for the addition he wants
to do to the code
```
  git checkout -b feature_DESCRIPTOR
```
This will initially be a copy of 'develop'.

4. Now David can do whatever work he wants to do on this branch.
```
  git add FILENAME
  ...
  git add -u             Good for safety; adds all tracked files
  git status             Check it looks OK
```
To submit changes to the feature branch remotely:
```
  git commit -m "type message here"
  git push -u origin feature_DESCRIPTOR
```
This last line has to be done once per local repository.  Next
time you can simply do
```
  git push origin
```
Note that any time you change any file, you need to "git add" it
again before committing and pushing.

5. If Sinisa did some work on the same branch, and then did
add/commit/push_origin David can add those changes by doing
```
  git pull -u origin feature_DESCRIPTOR
```
6. When David is done working on the feature branch, and it has
been pushed to origin, he should create a pull request to have
it merged into the 'develop' branch.  On the GitHub website.

  Click 'New pull request'

  In pulldown menus: Set
    base:      develop  <<--- ESSENTIAL PART: use develop here!
    compare:   feature_DESCRIPTOR
  
  Write comments
  
  Click 'Create pull request'

(See https://help.github.com/articles/creating-a-pull-request .)

As long as there are no conflicts, Sinisa does the merge into
'develop' on the Github website.
If there are conflicts then they need to be resolved in the 
terminal.

If Sinisa or David don't like something in the pull request, say
thay discuss something during pull request via website and they 
decide that further changes are needed to the branch, then they 
can simply git "add,commit,push" more changes to the origin via
terminal.  These additions should get updated in the github's
pull request on the website.

7. The feature_DESCRIPTOR branch can now be deleted:
```
  git branch -d feature_DESCRIPTOR                  ! locally
  git push origin --delete feature_DESCRIPTOR       ! remotely
```
8. When we have decided to issue a new release of the code we should
follow steps described in NEWVERSION
