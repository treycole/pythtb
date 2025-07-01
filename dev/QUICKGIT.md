*******************************************************

## Adding a feature
```bash
git checkout develop
git checkout -b feature_DESCRIPTOR
git add FILENAME
git commit -m "type message here"
git push -u origin feature_DESCRIPTOR
```

## Committing changes 

For every following commit, do

```bash
git add FILENAME
git commit -m "type message here"
git push origin
```

## Merging into `develop` 

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

## Remove branch
```bash
git branch -d feature_DESCRIPTOR
git push origin --delete feature_DESCRIPTOR
```

*******************************************************
